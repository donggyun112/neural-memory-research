"""Is there block structure in KC-to-MBON, or is there nothing for a task to use?

Phase 38 built a valence task and all three wirings agreed to three decimals.
That was a design fault — the signs were indexed by MBON column and rewiring
only moves which Kenyon cell reaches which MBON — but it also raised a prior
question that no task can answer: is there structure there at all?

The claim worth testing is specific. Compartments are the unit of learning: each
is innervated by its own dopaminergic neurons and read by its own MBONs, so a
lesson delivered to one should leave the others alone. That requires Kenyon cells
to be selective about which compartments they reach. If instead every cell spans
every compartment, one lesson smears across all of them, and the anatomy offers
no separation for a task to exploit.

So this measures the wiring directly, with no learning and no task: how
concentrated is a Kenyon cell's output across compartments, and how much do two
cells' compartment sets overlap — against the same curveball rewiring the other
probes use, which holds every degree fixed and moves only partners.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from wiring_prior_probe import rewire_preserving_degree

COMPARTMENT = re.compile(r"^(a'?[123]|b'?[12]|g[1-5]|CA|PED)\((L|R)\)$")


def mbon_compartments(neurons: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """The compartment each MBON reads, as an index into a name list."""
    mbons = neurons[neurons["type"].astype(str).str.startswith("MBON")]
    names: list[str] = []
    assigned = []
    for payload in mbons["roiInfo"]:
        regions = json.loads(payload) if isinstance(payload, str) else (payload or {})
        best, most = None, 0
        for name, info in regions.items():
            match = COMPARTMENT.match(name)
            if match and isinstance(info, dict) and (info.get("post") or 0) > most:
                best, most = match.group(1), info["post"]
        if best not in names:
            names.append(best)
        assigned.append(names.index(best))
    return np.array(assigned), names


def compartment_profile(weights: np.ndarray, compartment: np.ndarray, count: int) -> np.ndarray:
    """Per Kenyon cell, how much synaptic weight lands in each compartment."""
    profile = np.zeros((len(weights), count))
    for index in range(count):
        profile[:, index] = weights[:, compartment == index].sum(axis=1)
    return profile


def concentration(profile: np.ndarray) -> np.ndarray:
    """Share of a cell's output going to its single strongest compartment.

    One over the number of compartments means the cell spreads evenly and
    nothing is addressable; one means it speaks to exactly one compartment.
    """
    total = profile.sum(axis=1)
    alive = total > 0
    return profile[alive].max(axis=1) / total[alive]


def reach(profile: np.ndarray) -> np.ndarray:
    """How many compartments a cell reaches at all."""
    return (profile > 0).sum(axis=1)[profile.sum(axis=1) > 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="data/kc_mbon_weights.npy")
    parser.add_argument("--neurons", default="data/neurons.parquet")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    real = np.load(args.weights).astype(np.float64)
    compartment, names = mbon_compartments(pd.read_parquet(args.neurons))
    count = len(names)
    print(f"{real.shape[0]} Kenyon cells, {real.shape[1]} MBONs, {count} compartments")
    print(f"compartments: {' '.join(names)}\n")

    measures: dict[str, dict[str, list[float]]] = {
        name: {"concentration": [], "reach": []} for name in ("real", "rewired", "random")
    }
    for repeat in range(args.repeats):
        draw = np.random.default_rng(args.seed + repeat)
        wirings = {
            "real": real,
            "rewired": rewire_preserving_degree(real, draw),
            "random": draw.permutation(real.ravel()).reshape(real.shape),
        }
        for name, matrix in wirings.items():
            profile = compartment_profile(matrix, compartment, count)
            measures[name]["concentration"].append(float(concentration(profile).mean()))
            measures[name]["reach"].append(float(reach(profile).mean()))
        if repeat == 0:
            # The real wiring is fixed, so one pass is the whole story for it.
            pass

    print(f"{'wiring':>10} {'top-compartment share':>22} {'compartments reached':>22}")
    summary: dict[str, dict[str, np.ndarray]] = {}
    for name, values in measures.items():
        summary[name] = {k: np.array(v) for k, v in values.items()}
        print(
            f"{name:>10} {summary[name]['concentration'].mean():22.4f}"
            f" {summary[name]['reach'].mean():22.2f}"
        )
    print(f"{'even spread':>10} {1 / count:22.4f} {count:22d}")

    print(f"\n{'comparison':>28} {'difference':>11} {'95% interval':>22}")
    for measure in ("concentration", "reach"):
        for control in ("rewired", "random"):
            gap = summary["real"][measure] - summary[control][measure]
            maker = np.random.default_rng(7)
            draws = [gap[maker.integers(0, len(gap), len(gap))].mean() for _ in range(3000)]
            low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
            mark = "resolved" if low > 0 or high < 0 else "NOT resolved"
            label = f"{measure}: real over {control}"
            print(f"{label:>28} {gap.mean():+11.4f}  [{low:+.4f}, {high:+.4f}] {mark}")


def demo() -> None:
    """Concentration must see a cell that speaks to one compartment only."""
    # Two cells, four MBONs, two compartments.
    compartment = np.array([0, 0, 1, 1])
    focused = np.array([[5.0, 5.0, 0.0, 0.0]])
    spread = np.array([[2.5, 2.5, 2.5, 2.5]])
    assert concentration(compartment_profile(focused, compartment, 2))[0] == 1.0
    assert concentration(compartment_profile(spread, compartment, 2))[0] == 0.5
    assert reach(compartment_profile(focused, compartment, 2))[0] == 1
    assert reach(compartment_profile(spread, compartment, 2))[0] == 2
    # A cell with no output at all must not be counted as perfectly focused.
    assert len(concentration(compartment_profile(np.zeros((1, 4)), compartment, 2))) == 0
    print("demo ok")


if __name__ == "__main__":
    main()
