"""Compare repair conditions on the same tasks, paired.

The earlier phases compared conditions on next-action likelihood, where every
condition sees every position and pairing is automatic. Agent runs are not like
that: a run can time out, crash, or leave the workspace in a state the suite
cannot even load. Dropping those per condition would compare different task sets
and call the difference an effect, so only tasks that every condition attempted
are counted, and the discard is reported rather than hidden.

Three endpoints, because it is not yet known which one has any variance. If the
agent fixes everything, `resolved` is a constant and says nothing while
`actions` still separates a direct repair from a long search. If the agent fixes
little, `resolved` is the honest headline and `actions` mostly measures how long
it flailed. `repaired` sits between them and is the one with the most room.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def interval(gap: np.ndarray, draws: int = 3000, seed: int = 7) -> tuple[float, float]:
    maker = np.random.default_rng(seed)
    samples = [gap[maker.integers(0, len(gap), len(gap))].mean() for _ in range(draws)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def load(paths: list[Path]) -> dict[str, dict[str, dict]]:
    """condition -> task -> trial."""
    runs: dict[str, dict[str, dict]] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            trial = json.loads(line)
            runs.setdefault(trial["condition"], {})[trial["task"]] = trial
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path, nargs="+", help="one or more outcome files")
    parser.add_argument("--baseline", default="none", help="condition every gain is measured from")
    parser.add_argument(
        "--control",
        default="random",
        help="second reference: isolates how items are chosen from having items at all",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = load(args.trials)
    if len(runs) < 2:
        raise RuntimeError(f"need at least two conditions, found {sorted(runs)}")
    shared = sorted(set.intersection(*(set(tasks) for tasks in runs.values())))
    if not shared:
        raise RuntimeError("the conditions share no task")
    dropped = {name: len(tasks) - len(shared) for name, tasks in runs.items()}
    print(f"{len(shared)} tasks attempted by every condition")
    if any(dropped.values()):
        print(f"dropped, per condition: {dropped}")

    columns = {
        name: {
            "resolved": np.array([float(runs[name][t]["resolved"]) for t in shared]),
            "repaired": np.array([float(runs[name][t]["repaired"]) for t in shared]),
            "actions": np.array([float(runs[name][t]["actions"]) for t in shared]),
            "cheated": np.array([float(runs[name][t]["touched_tests"]) for t in shared]),
        }
        for name in sorted(runs)
    }

    print(f"\n{'condition':>14} {'resolved':>9} {'repaired':>9} {'actions':>8} {'edited tests':>13}")
    for name, values in columns.items():
        print(
            f"{name:>14} {values['resolved'].mean():9.3f} {values['repaired'].mean():9.2f}"
            f" {values['actions'].mean():8.1f} {int(values['cheated'].sum()):13d}"
        )

    # Fewer actions is better, so its sign is flipped to keep "positive is an
    # improvement" true for every row of the table below.
    endpoints = (("resolved", 1.0), ("repaired", 1.0), ("actions", -1.0))
    report: dict[str, dict] = {}
    for endpoint, sign in endpoints:
        print(f"\n{endpoint}: positive favours the first condition")
        if args.baseline not in columns:
            pairs = list(combinations(columns, 2))
        else:
            pairs = [(other, args.baseline) for other in columns if other != args.baseline]
            # Against `none` a condition is credited for the notes existing at
            # all — more text in the prompt, whatever it says. Against `random`
            # the text is held constant and only the choice of lines differs,
            # which is the quantity every selection rule claims to improve.
            pairs += [
                (other, args.control)
                for other in columns
                if other not in (args.control, args.baseline)
            ]
        for treatment, control in pairs:
            gap = sign * (columns[treatment][endpoint] - columns[control][endpoint])
            low, high = interval(gap)
            mark = "resolved" if low > 0 or high < 0 else "NOT resolved"
            label = f"{treatment} over {control}"
            report[f"{endpoint}:{label}"] = {
                "difference": float(gap.mean()),
                "low": low,
                "high": high,
            }
            print(f"  {label:>34} {gap.mean():+8.3f}  [{low:+.3f}, {high:+.3f}] {mark}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "tasks": shared,
                    "means": {
                        name: {key: float(values.mean()) for key, values in column.items()}
                        for name, column in columns.items()
                    },
                    "comparisons": report,
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"\nwrote {args.output}")


def demo() -> None:
    """Only tasks every condition attempted may be compared."""
    runs = {
        "none": {"a": {}, "b": {}, "c": {}},
        "spread": {"a": {}, "b": {}},
    }
    shared = sorted(set.intersection(*(set(tasks) for tasks in runs.values())))
    assert shared == ["a", "b"], shared
    # A condition that crashed on task c must not be credited with the tasks it
    # never attempted, nor compared against a baseline that did attempt them.
    assert len(runs["none"]) - len(shared) == 1
    gap = np.array([0.4, 0.5, 0.6, 0.5])
    low, high = interval(gap)
    assert low > 0 and high > low
    print("demo ok")


if __name__ == "__main__":
    main()
