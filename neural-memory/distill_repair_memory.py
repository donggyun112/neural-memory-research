"""Turn round one's raw tool calls into what was learned, not what was typed.

Every selection rule compared so far picks lines out of the log verbatim, so the
best any of them can do is hand over twelve tool calls. Read one and the limit is
obvious: `Read tests/test_ranges.py` is a fact about a file the next task does
not touch. The single line that could transfer — piping a 62,434-test suite
through `tail` — is buried among eleven that cannot, and similarity is the worst
offender because the nearest neighbours of a cue resemble each other and it
spends ten of its twelve slots on near-duplicates.

This is the condition those rules cannot express: one pass over the successful
runs that writes down the procedure rather than the transcript. It is still built
only from what round one actually did, and still never sees round two's modules.
The comparison it enables is the one worth having — whether the memory needs to
abstract, or whether choosing well among raw actions is enough.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from build_repair_memory import load_actions

BRIEF = """Below are the tool calls several agents made while fixing bugs in a Python \
repository. Each worked on a different file from the one you will be told about later, so \
nothing specific to those files is useful.

Write at most {keep} lines of procedural advice for an agent about to fix a bug in the same \
repository under a tight budget of turns. Each line must be an instruction it can act on, \
drawn from what these transcripts show worked. Prefer concrete commands over general advice. \
No preamble, no numbering, one instruction per line starting with "- ".

Transcripts:
{actions}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--per-task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=12)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--all-runs", action="store_true")
    args = parser.parse_args()

    trials = [json.loads(line) for line in args.trials.read_text().splitlines() if line.strip()]
    actions = load_actions(args.workspace, trials, solved_only=not args.all_runs)
    if not actions:
        raise RuntimeError("no actions to distil")

    brief = BRIEF.format(
        keep=args.keep,
        actions="\n".join(f"{task}: {text}" for task, text in actions),
    )
    finished = subprocess.run(
        f"claude -p {shlex.quote(brief)} --model {args.model} --safe-mode",
        shell=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    lines = [line.strip() for line in finished.stdout.splitlines() if line.strip().startswith("-")]
    if not lines:
        raise RuntimeError(f"the distiller returned nothing usable: {finished.stdout[:300]!r}")
    block = "\n".join(lines[: args.keep]) + "\n"

    # Written per task like every other condition, so the trial runner reads all
    # of them the same way; unlike similarity, the block does not depend on a cue.
    targets = [json.loads(line) for line in args.per_task.read_text().splitlines() if line.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    for task in targets:
        (args.output / f"{task['name']}.txt").write_text(block)
    print(f"distilled {len(actions)} actions into {len(lines[: args.keep])} lines\n{block}")


def demo() -> None:
    """Only lines that read as instructions survive the distiller's output."""
    raw = "Here is my advice:\n- Run the failing test alone first\nSome trailing chatter\n- Pipe output through tail\n"
    lines = [line.strip() for line in raw.splitlines() if line.strip().startswith("-")]
    assert lines == ["- Run the failing test alone first", "- Pipe output through tail"], lines
    print("demo ok")


if __name__ == "__main__":
    main()
