"""One table across the budgets, because a single budget could not say anything.

At 25 turns every condition repaired every task and at 12 nearly every one did,
so the binary endpoint was a constant and nothing could resolve. That is not
three failed experiments; it is a difficulty calibration whose answer is that the
round two modules are easy and the budget has to bind before the measurement has
any variance at all.

Laid side by side the budgets also carry their own claim. A memory of past
actions can only shorten a search, so it should do nothing when there is time to
spare and everything it can do when there is not. Whether the conditions separate
as the budget tightens is the prediction, and it is visible only across rows.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from analyze_repair_trials import interval, load

ORDER = ("none", "random", "spread", "similarity")


def budget_of(path: Path) -> str:
    """The turn budget a result file was produced under."""
    found = re.search(r"budget(\d+)\.", path.name)
    return found.group(1) if found else "25"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    by_budget: dict[str, list[Path]] = {}
    for path in args.trials:
        by_budget.setdefault(budget_of(path), []).append(path)

    print(f"{'budget':>7} " + " ".join(f"{name:>11}" for name in ORDER))
    rates: dict[str, dict[str, float]] = {}
    tables: dict[str, dict] = {}
    for budget in sorted(by_budget, key=int, reverse=True):
        runs = load(by_budget[budget])
        shared = sorted(set.intersection(*(set(t) for t in runs.values())))
        resolved = {
            name: np.array([float(runs[name][task]["resolved"]) for task in shared])
            for name in ORDER
            if name in runs
        }
        rates[budget] = {name: float(v.mean()) for name, v in resolved.items()}
        print(
            f"{budget:>5}t  "
            + " ".join(f"{rates[budget].get(name, float('nan')):>11.3f}" for name in ORDER)
            + f"   (n={len(shared)})"
        )
        # Against `random` the prompt carries the same amount of text and only
        # the choice of lines differs, which is what a selection rule claims.
        tables[budget] = {}
        for treatment in ("spread", "similarity"):
            if treatment not in resolved or "random" not in resolved:
                continue
            gap = resolved[treatment] - resolved["random"]
            low, high = interval(gap)
            tables[budget][f"{treatment} over random"] = {
                "difference": float(gap.mean()),
                "low": low,
                "high": high,
            }

    print(f"\n{'comparison':>28} {'budget':>7} {'difference':>11} {'95% interval':>22}")
    for budget, rows in tables.items():
        for label, row in rows.items():
            mark = "resolved" if row["low"] > 0 or row["high"] < 0 else "NOT resolved"
            print(
                f"{label:>28} {budget:>6}t {row['difference']:+11.3f}"
                f"  [{row['low']:+.3f}, {row['high']:+.3f}] {mark}"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"resolution": rates, "against_random": tables}, indent=2, sort_keys=True)
        )
        print(f"\nwrote {args.output}")


def demo() -> None:
    """A budget must be read from the filename, with the unlabelled one the first run."""
    assert budget_of(Path("round2.none.jsonl")) == "25"
    assert budget_of(Path("budget12.spread.jsonl")) == "12"
    assert budget_of(Path("budget8.similarity.jsonl")) == "8"
    print("demo ok")


if __name__ == "__main__":
    main()
