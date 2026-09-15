"""Give a repair task to an agent, and check whether the tests went green.

This is the endpoint phase 74 said was missing. Scoring the likelihood of a
recorded action cannot tell a better agent from a better predictor of an agent,
because both raise it. Running the agent and checking the suite can: a fix that
does not fix anything scores zero no matter how plausible it looked.

The harness is deliberately thin. It writes a task into an isolated copy of the
repository, hands the copy to a command, and runs the suite afterwards. What the
command is — an agent with memory, the same agent without it, or a control that
does nothing — is the experiment; this file only keeps the outcome honest.

Two things it refuses to leave to the agent's own account: whether the tests
pass, and whether the fix is the real one. The suite decides the first. For the
second the harness checks the mutated line changed, because an agent that edits
the *test* until it passes has not repaired anything.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROMPT = """A recent change broke this repository's test suite.

Repository root: {repo}
{report}

Find the defect in the source and fix it, then confirm the suite is green.

Rules:
- The bug is in the package source, not in the tests. Do not edit any test.
- Run the suite with: uv run --with pytest --with pretend python -m pytest tests -q
{memory}"""

# The most informative field of a tool call, in the order worth reporting.
FIELDS = ("command", "file_path", "pattern", "path", "query", "prompt", "url")


@dataclass
class Trial:
    """What one agent run on one task achieved."""

    task: str
    condition: str
    resolved: bool
    broken_before: int
    broken_after: int
    repaired: int
    touched_source: bool
    touched_tests: bool
    seconds: float
    actions: int


def tool_calls(transcript: Path, limit: int = 240) -> list[str]:
    """Every tool call the agent actually made, from its own event stream.

    An earlier version asked the agent to append its calls to a log. It
    under-reported badly — one run worked for 144 seconds and logged two
    actions — so the stream is parsed instead. The agent cannot mis-state what
    it did here, because this is the record of it doing it.
    """
    if not transcript.exists():
        return []
    calls: list[str] = []
    for line in transcript.read_text(errors="replace").splitlines():
        if '"tool_use"' not in line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for block in event.get("message", {}).get("content", []) or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            payload = block.get("input") or {}
            argument = next(
                (str(payload[field]) for field in FIELDS if payload.get(field)),
                "",
            )
            calls.append(f"{block.get('name', '?')} {argument}".strip()[:limit])
    return calls


def prepare(source: Path, workspace: Path, task: dict) -> Path:
    """A fresh copy of the repository with the defect applied."""
    if workspace.exists():
        shutil.rmtree(workspace)
    # The git history holds the original line, which would turn the task into a
    # lookup rather than a repair.
    shutil.copytree(source, workspace, ignore=shutil.ignore_patterns(".git"))
    target = workspace / task["path"]
    lines = target.read_text().splitlines(keepends=True)
    ending = "\n" if lines[task["line"] - 1].endswith("\n") else ""
    lines[task["line"] - 1] = task["after"] + ending
    target.write_text("".join(lines))
    return workspace


def failing_tests(repo: Path, tests: str, timeout: int) -> tuple[int, bool]:
    try:
        finished = subprocess.run(
            [
                "uv", "run", "--with", "pytest", "--with", "pretend",
                "python", "-m", "pytest", tests,
                "-q", "--tb=no", "-p", "no:cacheprovider", "--no-header",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return -1, False
    return len(re.findall(r"^(?:FAILED|ERROR) \S+", finished.stdout, flags=re.M)), True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True, help="the clean repository")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--agent",
        required=True,
        help="command run per task; {prompt} and {repo} are substituted",
    )
    parser.add_argument("--condition", required=True, help="label for this arm")
    parser.add_argument("--memory", type=Path, help="text pasted into the prompt")
    parser.add_argument("--tests", default="tests")
    parser.add_argument(
        "--blind",
        action="store_true",
        help="withhold the failing test names, leaving only that the suite is red",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 runs every task")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--agent-timeout", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    if args.limit:
        tasks = tasks[: args.limit]
    memory = f"\nNotes from earlier repairs:\n{args.memory.read_text()}" if args.memory else ""

    trials: list[Trial] = []
    for position, task in enumerate(tasks, start=1):
        workspace = prepare(args.repo, args.workspace / task["name"], task)
        transcript = workspace / "agent-transcript.jsonl"
        # Naming the failing tests points straight at the module and makes the
        # task solvable without searching, which leaves no room for a memory to
        # help. Blind mode reports only that the suite is red.
        report = (
            f"{task['broken']} tests are failing."
            if args.blind
            else "Failing tests ({broken} total, up to 10 shown):\n{failing}".format(
                broken=task["broken"],
                failing="\n".join(f"  {node}" for node in task["failing"][:10]),
            )
        )
        prompt = PROMPT.format(repo=workspace, report=report, memory=memory)
        started = time.time()
        with transcript.open("w") as stream:
            try:
                subprocess.run(
                    args.agent.format(prompt=prompt, repo=workspace),
                    shell=True,
                    cwd=workspace,
                    stdout=stream,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=args.agent_timeout,
                )
            except subprocess.TimeoutExpired:
                pass
        elapsed = time.time() - started
        calls = tool_calls(transcript)

        after, complete = failing_tests(workspace, args.tests, args.timeout)
        edited = (workspace / task["path"]).read_text().splitlines()
        # An agent that rewrites the test until it passes has not repaired
        # anything, so the outcome records both edits and the scoring can
        # exclude them.
        # Running the suite leaves __pycache__ under tests/, which a plain
        # recursive diff reports as a difference and which would mark every
        # honest run as cheating.
        tests_changed = subprocess.run(
            [
                "diff", "-rq",
                "-x", "__pycache__", "-x", "*.pyc", "-x", ".pytest_cache",
                str(args.repo / args.tests), str(workspace / args.tests),
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        trials.append(
            Trial(
                task=task["name"],
                condition=args.condition,
                resolved=complete and after == 0 and not tests_changed,
                broken_before=task["broken"],
                broken_after=after,
                repaired=max(0, task["broken"] - after) if complete else 0,
                touched_source=edited[task["line"] - 1].strip() != task["after"].strip(),
                touched_tests=bool(tests_changed),
                seconds=elapsed,
                actions=len(calls),
            )
        )
        (workspace / "agent-actions.log").write_text("".join(f"{c}\n" for c in calls))
        mark = "fixed" if trials[-1].resolved else f"{after} still failing"
        print(
            f"[{position:>3}/{len(tasks)}] {task['name']:<24} {mark:<18}"
            f" {elapsed:5.0f}s  {trials[-1].actions:>3} actions",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(asdict(t)) + "\n" for t in trials))
    solved = sum(t.resolved for t in trials)
    cheated = sum(t.touched_tests for t in trials)
    print(f"\n{args.condition}: {solved}/{len(trials)} resolved, {cheated} edited tests")
    print(f"wrote {args.output}")


def demo() -> None:
    """A run that edits the tests must not count as a repair."""
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        transcript = Path(folder) / "t.jsonl"
        transcript.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "looking"},
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "input": {"command": "pytest tests -q"},
                            },
                        ]
                    },
                }
            )
            + "\nnot json at all\n"
        )
        # Text blocks are not calls, and a malformed line must not lose the rest.
        assert tool_calls(transcript) == ["Bash pytest tests -q"], tool_calls(transcript)
        assert tool_calls(Path(folder) / "missing.jsonl") == []

    trial = Trial(
        task="version-12",
        condition="check",
        resolved=False,
        broken_before=3,
        broken_after=0,
        repaired=3,
        touched_source=False,
        touched_tests=True,
        seconds=1.0,
        actions=4,
    )
    # Zero failures is not enough: the suite can be made green by deleting the
    # test that noticed, which is why `resolved` also requires the tests intact.
    assert trial.broken_after == 0 and not trial.resolved
    print("demo ok")


if __name__ == "__main__":
    main()
