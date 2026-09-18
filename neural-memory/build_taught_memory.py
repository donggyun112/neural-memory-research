"""Write-time selection: teach from failure instead of discarding it.

`build_repair_memory.py` and `distill_repair_memory.py` both default to
`solved_only`, and the latter's docstring for why is exactly phase 74's own
finding: a log of an agent failing is not a note worth passing on. That throws
away every action from an unresolved run before any selection rule sees it.

The fly does not discard the punished trial, it teaches from it — the same
`teach()` call runs on outcome -1 as on +1, just with the sign flipped. This is
the agent-memory analogue: distill the successful runs into what worked, same
as `distill_repair_memory.py`, and separately distill the failed runs into
named dead ends, then hand a round-two agent both. A raw failed-run action
carries no signal by itself (a `Read` or a `Bash pytest` call looks the same
whether the run that contained it succeeded or not) — only the whole failed
trajectory, synthesised by something that saw how it ended, can say "this did
not work." That is why this goes through the distiller rather than quoting
lines the way `spread` / `random` / `similarity` do.

Falsification is stated in board/003-write-time-selection.md, not here: this
file only builds the note, it does not decide whether the note helped.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from build_repair_memory import load_actions

DO_BRIEF = """Below are the tool calls several agents made while successfully fixing bugs in \
a Python repository. Each worked on a different file from the one you will be told about later, \
so nothing specific to those files is useful.

Write at most {keep} lines of procedural advice for an agent about to fix a bug in the same \
repository under a tight budget of turns. Each line must be an instruction it can act on, drawn \
from what these transcripts show worked. Prefer concrete commands over general advice. No \
preamble, no numbering, one instruction per line starting with "- ".

Transcripts:
{actions}"""

AVOID_BRIEF = """Below are the tool calls several agents made while attempting to fix bugs in a \
Python repository. Every one of these runs FAILED: the suite was still red when the agent \
stopped or ran out of turns. Each worked on a different file from the one you will be told about \
later, so nothing specific to those files is useful.

Write at most {keep} lines naming specific dead ends or mistakes to avoid, drawn from what these \
failed attempts show did not work. Each line must name a concrete action, assumption, or approach \
that turned out to be unproductive -- not generic advice that would apply to any bug. If nothing \
in the transcripts points to a specific reusable dead end, write fewer lines rather than padding \
with generic caution. No preamble, no numbering, one instruction per line starting with "- avoid ".

Transcripts:
{actions}"""


def distil(brief: str, keep: int, actions: list[tuple[str, str]], model: str) -> list[str]:
    if not actions or keep <= 0:
        return []
    prompt = brief.format(
        keep=keep, actions="\n".join(f"{task}: {text}" for task, text in actions)
    )
    finished = subprocess.run(
        f"claude -p {shlex.quote(prompt)} --model {model} --safe-mode",
        shell=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    lines = [line.strip() for line in finished.stdout.splitlines() if line.strip().startswith("-")]
    if not lines:
        raise RuntimeError(f"the distiller returned nothing usable: {finished.stdout[:300]!r}")
    return lines[:keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--per-task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=12, help="total lines, split do/avoid")
    parser.add_argument("--avoid-fraction", type=float, default=1 / 3)
    parser.add_argument("--model", default="sonnet")
    args = parser.parse_args()

    trials = [json.loads(line) for line in args.trials.read_text().splitlines() if line.strip()]
    solved = [trial for trial in trials if trial["resolved"]]
    failed = [trial for trial in trials if not trial["resolved"]]

    do_actions = load_actions(args.workspace, solved, solved_only=True)
    avoid_actions = load_actions(args.workspace, failed, solved_only=False)
    if not do_actions:
        raise RuntimeError("no solved-run actions to distil -- nothing worked in round one")

    avoid_keep = round(args.keep * args.avoid_fraction) if avoid_actions else 0
    do_keep = args.keep - avoid_keep

    do_lines = distil(DO_BRIEF, do_keep, do_actions, args.model)
    avoid_lines = distil(AVOID_BRIEF, avoid_keep, avoid_actions, args.model) if avoid_keep else []

    print(f"{len(solved)} solved / {len(failed)} failed round-one trials")
    print(f"do ({len(do_lines)} lines, from {len(do_actions)} actions):")
    print("\n".join(do_lines))
    print(f"\navoid ({len(avoid_lines)} lines, from {len(avoid_actions)} actions):")
    print("\n".join(avoid_lines) if avoid_lines else "(none -- no failed round-one trials)")

    block = "What worked before:\n" + "\n".join(do_lines) + "\n"
    if avoid_lines:
        block += "\nWhat did not work before (avoid these):\n" + "\n".join(avoid_lines) + "\n"

    targets = [json.loads(line) for line in args.per_task.read_text().splitlines() if line.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    for task in targets:
        (args.output / f"{task['name']}.txt").write_text(block)
    print(f"\nwrote {len(targets)} note blocks to {args.output}/")


def demo() -> None:
    """Failed-run actions must not silently vanish when there are none to distil."""
    assert distil(AVOID_BRIEF, 0, [("t", "Read x.py")], "sonnet") == []
    assert distil(AVOID_BRIEF, 4, [], "sonnet") == []
    print("demo ok")


if __name__ == "__main__":
    main()
