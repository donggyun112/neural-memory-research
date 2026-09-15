"""Partition repair tasks so no source file appears in both rounds.

Round one produces the memory and round two is where it is tested. If both
rounds draw on the same module, a round two agent handed notes from round one is
being handed notes about the very file its bug is in — which measures whether
reading that file twice helps, not whether a memory does. The mutation sites
cluster heavily (a few large modules supply most of them), so this is the
default outcome rather than an unlucky one.

Splitting by file makes the transfer the thing under test: whatever round one
learned has to be general enough to survive the move to code it never saw.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def partition(tasks: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split into (round one, round two) with no module shared.

    Whole modules move together, so the round sizes land near the requested
    fraction rather than on it.
    """
    by_module: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        by_module[task["path"]].append(task)
    modules = sorted(by_module)
    random.Random(seed).shuffle(modules)

    wanted = fraction * len(tasks)
    first: list[str] = []
    second: list[str] = []
    taken = 0
    for module in modules:
        # The budget is counted in tasks, not modules. Comparing a module count
        # against a task count sends every module to the first round and leaves
        # the second empty, which is what this did until it was run.
        #
        # Largest-remaining would balance better, but the shuffle has to stay the
        # only thing deciding which module lands where, or the split correlates
        # with module size and so with how well covered the module is.
        if taken < wanted:
            first.append(module)
            taken += len(by_module[module])
        else:
            second.append(module)
    return (
        [task for module in first for task in by_module[module]],
        [task for module in second for task in by_module[module]],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True, help="round one, builds the memory")
    parser.add_argument("--second", type=Path, required=True, help="round two, tests it")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    first, second = partition(tasks, args.fraction, args.seed)
    for path, part in ((args.first, first), (args.second, second)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(task) + "\n" for task in part))
    overlap = {t["path"] for t in first} & {t["path"] for t in second}
    assert not overlap, overlap
    print(
        f"round one: {len(first)} tasks in {len({t['path'] for t in first})} modules\n"
        f"round two: {len(second)} tasks in {len({t['path'] for t in second})} modules"
    )


def demo() -> None:
    """Both rounds must be non-empty and share no module.

    The first version of this check asserted only that the rounds do not overlap
    and that the first is non-empty. Sending every task to the first round
    satisfies both, so the check passed while the split was useless. Asserting
    the second round is populated is the assertion that had to be here.
    """
    tasks = [
        {"name": f"{module}-{index}", "path": module}
        for module, count in (("a.py", 20), ("b.py", 5), ("c.py", 8), ("d.py", 1))
        for index in range(count)
    ]
    for fraction in (0.2, 0.5, 0.8):
        first, second = partition(tasks, fraction, seed=3)
        assert len(first) + len(second) == len(tasks)
        assert not {t["path"] for t in first} & {t["path"] for t in second}
        assert first and second, (fraction, len(first), len(second))
    # Whole modules move together, so a dominant module pulls the split away
    # from the requested fraction; the caller reads the printed counts.
    first, second = partition(tasks, 0.5, seed=3)
    assert len(first) + len(second) == 34
    print("demo ok")


if __name__ == "__main__":
    main()
