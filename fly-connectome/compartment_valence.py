"""Read each mushroom-body compartment's valence sign off the connectome.

The previous probe scored the KC-to-MBON wiring on linear associative capacity
and found it no better than a degree-preserving rewiring, and worse than a
uniform shuffle. The literature says why: random expansion maximises capacity,
and the real wiring pays capacity for selectivity, biasing toward what matters
to the animal. Capacity is simply not what this wiring is for.

What it is for is compartments. Dopaminergic neurons tile the lobes, and the two
families carry opposite meanings — PAM largely reward, PPL1 largely punishment.
Which family innervates a compartment therefore says what that compartment's
KC-to-MBON synapses learn to signal, and that is readable from the anatomy alone
with no training and no behavioural data.

This extracts it: per compartment, the presynapse counts of each family and the
sign their ratio implies. Nothing here learns. It is a measurement of the wiring,
and the number it produces is the thing the capacity probe was missing.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

# Compartments of the lobes plus the peduncle and calyx. The side suffix is
# stripped so left and right are pooled: they are the same compartment.
COMPARTMENT = re.compile(r"^(a'?[123]|b'?[12]|g[1-5]|CA|PED)\((L|R)\)$")


def compartment_presynapses(neurons: pd.DataFrame) -> pd.DataFrame:
    """Presynapse counts per (compartment, dopaminergic family)."""
    kinds = neurons["type"].astype(str).str.extract(r"^(PAM|PPL)")[0]
    rows: list[dict] = []
    for kind, payload in zip(kinds, neurons["roiInfo"]):
        if not isinstance(kind, str):
            continue
        regions = json.loads(payload) if isinstance(payload, str) else (payload or {})
        for name, info in regions.items():
            match = COMPARTMENT.match(name)
            if not match or not isinstance(info, dict):
                continue
            # Presynapses are where the neuron releases, so they are what decides
            # which compartment a dopaminergic signal is delivered to.
            release = info.get("pre") or 0
            if release > 0:
                rows.append({"compartment": match.group(1), "family": kind, "pre": release})
    counted = pd.DataFrame(rows)
    return counted.pivot_table(
        index="compartment", columns="family", values="pre", aggfunc="sum", fill_value=0
    )


def valence(table: pd.DataFrame, floor: int = 50) -> pd.DataFrame:
    """Signed index per compartment: +1 all reward, -1 all punishment.

    A compartment with almost no dopaminergic input has a ratio that is noise, so
    `floor` marks those unresolved rather than letting a handful of synapses
    decide a sign.
    """
    reward = table.get("PAM", pd.Series(0, index=table.index)).astype(float)
    punish = table.get("PPL", pd.Series(0, index=table.index)).astype(float)
    total = reward + punish
    index = np.where(total > 0, (reward - punish) / total.clip(lower=1), 0.0)
    return pd.DataFrame(
        {
            "PAM": reward.astype(int),
            "PPL": punish.astype(int),
            "valence": index,
            "resolved": total >= floor,
        }
    ).sort_values("valence", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", default="data/neurons.parquet")
    parser.add_argument("--floor", type=int, default=50, help="minimum presynapses to call a sign")
    parser.add_argument("--output", default="data/compartment_valence.csv")
    args = parser.parse_args()

    neurons = pd.read_parquet(args.neurons)
    table = valence(compartment_presynapses(neurons), args.floor)
    print(f"{'compartment':>12} {'PAM':>8} {'PPL':>8} {'valence':>9}  sign")
    for name, row in table.iterrows():
        if not row["resolved"]:
            sign = "unresolved"
        else:
            sign = "reward" if row["valence"] > 0.33 else (
                "punishment" if row["valence"] < -0.33 else "mixed"
            )
        print(
            f"{name:>12} {int(row['PAM']):>8} {int(row['PPL']):>8}"
            f" {row['valence']:>9.3f}  {sign}"
        )
    table.to_csv(args.output)
    print(f"\nwrote {args.output}")


def demo() -> None:
    """A sign must follow the dopaminergic majority, and thin evidence must abstain."""
    table = pd.DataFrame(
        {"PAM": [900, 10, 400, 5], "PPL": [10, 900, 400, 2]},
        index=["g5", "g1", "b2", "a3"],
    )
    got = valence(table, floor=50)
    assert got.loc["g5", "valence"] > 0.9, got.loc["g5", "valence"]
    assert got.loc["g1", "valence"] < -0.9, got.loc["g1", "valence"]
    assert abs(got.loc["b2", "valence"]) < 0.01, "an even split is not a sign"
    # Seven synapses must not be allowed to name a compartment's meaning.
    assert not got.loc["a3", "resolved"]
    assert COMPARTMENT.match("g5(R)") and not COMPARTMENT.match("LAL(R)")
    print("demo ok")


if __name__ == "__main__":
    main()
