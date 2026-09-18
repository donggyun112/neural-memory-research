"""What a run spent its turns on, so two pools can be compared rather than described.

Phase 86 asked whether failed repair runs contain dead ends — hypotheses formed,
acted on, and found wrong — because the write-time-selection idea depends on
there being some. On the local pool the answer was decisive: eleven failures,
zero edits, all of them opening by locating files. Nothing was tried, so nothing
was tried and rejected.

The same question has to be asked of a cross-module pool, where locating the
defect *is* hypothesis formation and a failure could plausibly take the shape an
avoid-line describes. Two sessions reading transcripts narratively would produce
two answers that agree or differ for reasons neither could name, so this is the
classifier itself rather than a description of one.

    locate  find, ls, tree, grep -r, wc — where is anything
    test    any pytest invocation
    read    Read of a file
    edit    Edit or Write — the only kind that commits to a hypothesis
    other   everything else
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

LOCATE = ("find ", "ls ", "tree ", "grep -r", "wc -l")


def kind(line: str) -> str:
    """One action, one label. Order matters: an edit is an edit whatever it says."""
    if line.startswith(("Edit", "Write")):
        return "edit"
    if line.startswith("Read"):
        return "read"
    lowered = line.lower()
    if "pytest" in lowered:
        return "test"
    if line.startswith("Bash") and any(mark in lowered for mark in LOCATE):
        return "locate"
    return "other"


def sequences(workspace: Path, outcomes: Path) -> list[tuple[str, bool, list[str]]]:
    """(task, resolved, action kinds) for every run with both a log and an outcome."""
    resolved = {
        json.loads(line)["task"]: json.loads(line)["resolved"]
        for line in outcomes.read_text().splitlines()
        if line.strip()
    }
    rows = []
    for log in sorted(workspace.glob("*/agent-actions.log")):
        task = log.parent.name
        if task not in resolved:
            continue
        rows.append(
            (task, resolved[task], [kind(l.strip()) for l in log.read_text().splitlines() if l.strip()])
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="where the runs left logs")
    parser.add_argument("--outcomes", type=Path, required=True, help="the trial jsonl")
    parser.add_argument("--label", default="", help="name for this pool in the output")
    args = parser.parse_args()

    rows = sequences(args.workspace, args.outcomes)
    if not rows:
        raise RuntimeError(f"no runs with both a log and an outcome under {args.workspace}")
    failed = [r for r in rows if not r[1]]
    passed = [r for r in rows if r[1]]
    print(f"{args.label or args.workspace.name}: {len(failed)} failed, {len(passed)} resolved\n")

    print(f"{'':10} {'actions':>8} " + " ".join(f"{k:>7}" for k in
                                                ("locate", "test", "read", "edit", "other")))
    for name, group in (("failed", failed), ("resolved", passed)):
        tally = Counter(k for _, _, seq in group for k in seq)
        total = sum(tally.values()) or 1
        print(
            f"{name:10} {total:8} "
            + " ".join(f"{tally[k] / total:6.0%} " for k in
                       ("locate", "test", "read", "edit", "other"))
        )

    # The percentages are decoration next to this: whether any failing run ever
    # committed to a hypothesis. An avoid-line needs one to name.
    with_edit = sum(1 for _, _, seq in failed if "edit" in seq)
    print(f"\nfailed runs that reached an edit: {with_edit} of {len(failed)}")
    if failed:
        print("\nfailed runs, one letter per action:")
        for task, _, seq in failed:
            print(f"  {task:<24} {' '.join(s[0] for s in seq)}")


def demo() -> None:
    """The labels have to survive the shapes these logs actually take."""
    assert kind("Edit /a/b/c.py") == "edit"
    assert kind("Write /a/b/c.py") == "edit"
    assert kind("Read /a/b/c.py") == "read"
    assert kind("Bash uv run --with pytest python -m pytest tests -q") == "test"
    assert kind("Bash find . -name '*.py' | head -20") == "locate"
    # A pytest command that also greps is a test, not a locate — the first match
    # wins, and running the suite is the more informative thing it did.
    assert kind('Bash pytest tests -q 2>&1 | grep -r FAILED') == "test"
    assert kind("Bash git log --oneline -10") == "other"
    print("demo ok")


if __name__ == "__main__":
    main()
