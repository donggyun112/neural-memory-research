"""Verifiable bug-fix tasks, made by breaking working code on purpose.

Everything measured through phase 77 scored the likelihood of an action a past
agent took. Phase 74 showed why that is not enough: memory raises that
likelihood just as much on trajectories that failed to fix the bug as on ones
that succeeded, so the number cannot separate a better agent from a better
predictor of an agent. Even the oracle cannot separate them.

The separation is free if the agent acts and the outcome is checked. That needs
tasks where success is a fact rather than a judgement, and enough of them to
resolve an interval. A mutation to one line of a covered function is exactly
that: the suite was green, the mutation makes it red, and the task is to make it
green again without being told where the edit was.

No docker and no benchmark harness — a pure-python repository with a fast suite
gives thousands of candidate sites and verifies in seconds.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Each rule rewrites one occurrence in a line. They are deliberately small: a
# flipped comparison or a dropped negation is the kind of defect a repair agent
# has to localise by reading test output, not by spotting mangled syntax.
MUTATIONS: list[tuple[str, str]] = [
    (r"(?<![=!<>])==(?!=)", "!="),
    (r"!=", "=="),
    (r"(?<![<>=!])<=", "<"),
    (r"(?<![<>=!])>=", ">"),
    (r"(?<![-<>=!])<(?![=<])", "<="),
    (r"(?<![-<>=!])>(?![=>])", ">="),
    (r"\band\b", "or"),
    (r"\bor\b", "and"),
    (r"\bTrue\b", "False"),
    (r"\bFalse\b", "True"),
    (r"\bnot\s+", ""),
    (r"\bmin\(", "max("),
    (r"\bmax\(", "min("),
    (r"\+ 1\b", "- 1"),
    (r"- 1\b", "+ 1"),
]

SKIP = re.compile(r"^\s*(#|\"\"\"|'''|from |import |@)|^\s*$")


@dataclass
class Task:
    """One broken line and the tests that notice."""

    name: str
    path: str
    line: int
    before: str
    after: str
    rule: str
    failing: list[str]
    broken: int


def candidate_lines(source: Path) -> list[tuple[int, str, str, str]]:
    """Every (line number, original, mutated, rule) this file admits."""
    found = []
    for number, text in enumerate(source.read_text().splitlines(), start=1):
        if SKIP.match(text):
            continue
        for pattern, replacement in MUTATIONS:
            mutated, count = re.subn(pattern, replacement, text, count=1)
            if count and mutated != text:
                found.append((number, text, mutated, f"{pattern} -> {replacement}"))
    return found


def parses(source: Path) -> bool:
    """Whether the mutated file is still valid Python.

    A rule that breaks syntax produces a collection error, not a logic defect,
    and an agent asked to repair it is doing a different task. `SKIP` keeps
    mutations off imports and comments for the same reason, but it guards the
    line a mutation lands on rather than what the mutation makes of it: `>` to
    `>=` on a return annotation yields `->= str:`, which SKIP cannot see coming.
    Checking the result covers every rule, including ones not written yet.
    """
    try:
        compile(source.read_text(), str(source), "exec")
    except SyntaxError:
        return False
    return True


def apply(source: Path, line: int, text: str) -> None:
    lines = source.read_text().splitlines(keepends=True)
    ending = "\n" if lines[line - 1].endswith("\n") else ""
    lines[line - 1] = text + ending
    source.write_text("".join(lines))


def run_tests(repo: Path, tests: str, timeout: int) -> tuple[list[str], bool]:
    """Node ids of failing tests, and whether the run completed at all.

    A mutation can make the suite hang rather than fail, so a timeout counts as
    unusable rather than as a very effective break.
    """
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
        return [], False
    failures = re.findall(r"^(FAILED|ERROR) (\S+)", finished.stdout, flags=re.M)
    return [node for _, node in failures], True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", default="src/packaging", help="package directory")
    parser.add_argument("--tests", default="tests")
    parser.add_argument("--count", type=int, default=80, help="tasks wanted")
    parser.add_argument("--attempts", type=int, default=400)
    parser.add_argument(
        "--min-broken", type=int, default=1, help="a mutation nothing notices is not a task"
    )
    parser.add_argument(
        "--max-broken",
        type=int,
        default=40,
        help="a mutation that breaks hundreds of tests names its own location",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--cross-module-only",
        action="store_true",
        help=(
            "keep only mutations whose failing tests do not live in a file named "
            "after the mutated module -- otherwise the agent's own first pytest "
            "run hands back the location blind/budget never withheld"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    sources = sorted((repo / args.source).rglob("*.py"))
    sites = [
        (path, *option)
        for path in sources
        if path.name != "__init__.py"
        for option in candidate_lines(path)
    ]
    if not sites:
        raise RuntimeError("no mutable line found")
    random.Random(args.seed).shuffle(sites)
    print(f"{len(sites)} candidate sites across {len(sources)} files", flush=True)

    baseline, complete = run_tests(repo, args.tests, args.timeout)
    if not complete or baseline:
        raise RuntimeError(f"the suite must be green before it can be broken: {baseline[:3]}")

    tasks: list[Task] = []
    started = time.time()
    for index, (path, line, before, after, rule) in enumerate(sites[: args.attempts]):
        if len(tasks) >= args.count:
            break
        apply(path, line, after)
        try:
            if not parses(path):
                continue
            failing, complete = run_tests(repo, args.tests, args.timeout)
        finally:
            apply(path, line, before)
        if not complete or not args.min_broken <= len(failing) <= args.max_broken:
            continue
        if args.cross_module_only and any(path.stem in node for node in failing):
            continue
        tasks.append(
            Task(
                # Two rules can fire on the same line — a comparison flip and a
                # boundary shift, say — so the line alone does not name a task.
                # Colliding names silently collapse into one when the outcomes
                # are keyed for pairing, and a condition quietly loses a task.
                name=f"{path.stem}-{line}-{len(tasks)}",
                path=str(path.relative_to(repo)),
                line=line,
                before=before,
                after=after,
                rule=rule,
                failing=sorted(failing)[:20],
                broken=len(failing),
            )
        )
        rate = (time.time() - started) / (index + 1)
        print(
            f"[{len(tasks):>3}/{args.count}] {path.name}:{line} breaks {len(failing)}"
            f"  ({rate:.0f}s per attempt)",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(asdict(task)) + "\n" for task in tasks))
    print(f"\nwrote {len(tasks)} tasks to {args.output}")


def demo() -> None:
    """The mutation rules must change a line without mangling neighbours."""
    cases = [
        ("if a == b:", "if a != b:"),
        ("while x <= y:", "while x < y:"),
        ("return not ready", "return ready"),
        ("value = min(a, b)", "value = max(a, b)"),
    ]
    for text, expected in cases:
        for pattern, replacement in MUTATIONS:
            mutated, count = re.subn(pattern, replacement, text, count=1)
            if count and mutated == expected:
                break
        else:
            raise AssertionError(f"no rule turns {text!r} into {expected!r}")
    # Imports and comments are never mutation sites: a repair task has to be a
    # logic defect, not a syntactically obvious edit.
    for skipped in ("import os", "# a == b", "    ", "from x import y"):
        assert SKIP.match(skipped), skipped
    # A return annotation is not a comparison. `>` -> `>=` on `-> str:` produced
    # `->= str:`, a collection error rather than a bug to find.
    for pattern, replacement in MUTATIONS:
        mutated, count = re.subn(pattern, replacement, "def f(self) -> str:", count=1)
        assert not count or "->=" not in mutated, (pattern, mutated)
    assert not SKIP.match("    if a == b:")
    print("demo ok")


if __name__ == "__main__":
    main()
