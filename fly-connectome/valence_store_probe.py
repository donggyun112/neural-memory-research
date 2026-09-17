"""Store good-or-bad, not arbitrary vectors, and ask if the wiring earns its keep.

The capacity probe stored random values in random output directions and found the
real KC-to-MBON wiring no better than a degree-preserving rewiring, and worse
than a uniform shuffle. That is the expected answer to the wrong question: a
uniform expansion maximises capacity, and the literature says the fly pays
capacity for selectivity.

What the wiring is organised for is compartments. `compartment_valence.py` reads
the sign of each one off the anatomy — PAM against PPL1 presynapses — and the
textbook split falls out with nothing fitted: horizontal lobes reward, vertical
lobes punish. So the 75 MBON columns are not interchangeable. Eight of them mean
"good" and eight mean "bad", and a rewiring that moves a synapse from a reward
compartment to a punishment one has changed what that synapse can say.

This stores exactly that: each pattern carries one bit, good or bad, and recall
is whether the readout's sign comes back right. Nothing in the capacity task
could see the difference, because a random target vector has no sign to get
wrong.

Falsification, fixed before running: if the real wiring does not beat the
degree-preserving rewiring on signed recall either, then the compartment
structure buys nothing a shuffled matrix of the same shape cannot do, and the
anatomy's organisation is not what makes valence learnable.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from wiring_prior_probe import rewire_preserving_degree, sparse_code

COMPARTMENT = re.compile(r"^(a'?[123]|b'?[12]|g[1-5]|CA|PED)\((L|R)\)$")


def mbon_signs(neurons: pd.DataFrame, valence_table: pd.DataFrame) -> np.ndarray:
    """One valence sign per MBON, from the compartment it reads.

    An MBON is assigned the compartment holding most of its postsynapses, which
    is where its dendrites sit among the Kenyon cell axons.
    """
    mbons = neurons[neurons["type"].astype(str).str.startswith("MBON")]
    signs = []
    for payload in mbons["roiInfo"]:
        regions = json.loads(payload) if isinstance(payload, str) else (payload or {})
        best, most = None, 0
        for name, info in regions.items():
            match = COMPARTMENT.match(name)
            if match and isinstance(info, dict) and (info.get("post") or 0) > most:
                best, most = match.group(1), info["post"]
        signs.append(float(valence_table["valence"].get(best, 0.0)) if best else 0.0)
    return np.array(signs)


def store_signed(
    mask: np.ndarray, codes: np.ndarray, labels: np.ndarray, signs: np.ndarray
) -> np.ndarray:
    """Depression written into the synapses, as a change from the resting drive.

    This is the fly's rule rather than a generic delta rule: dopamine arriving in
    a compartment depresses the KC-to-MBON synapses active at that moment, and
    which compartment it arrives in is set by whether the outcome was good or
    bad. A synapse can only learn about the valence its compartment carries,
    which is the structure the capacity task could not see.

    Only the change is returned. Returning the whole matrix buries it: the
    resting strengths sum to 229,528 while a lesson moves a synapse by about one,
    so the constant swamped the signal and every wiring scored chance.
    """
    support = (mask != 0).astype(np.float64)
    depression = np.zeros_like(support)
    for code, label in zip(codes, labels):
        # A good outcome drives the reward compartments' dopamine, depressing the
        # synapses onto reward-reading MBONs, so what survives leans the other
        # way. The sign a synapse can learn is fixed by where it sits.
        depression += np.outer(code, label * signs) * support
    return depression


def recall_sign(
    depression: np.ndarray, codes: np.ndarray, labels: np.ndarray, signs: np.ndarray
) -> float:
    """Share of patterns whose recalled valence has the right sign.

    The ensemble votes weighted by what each MBON means: reward-compartment
    output counts positive, punishment-compartment output negative. An unweighted
    sum would treat all 75 columns as interchangeable, which is the assumption
    the compartment structure exists to deny.
    """
    read = codes @ depression
    return float((np.sign(read @ signs) == labels).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="data/kc_mbon_weights.npy")
    parser.add_argument("--neurons", default="data/neurons.parquet")
    parser.add_argument("--valence", default="data/compartment_valence.csv")
    parser.add_argument("--items", type=int, default=400)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--active", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    real = np.load(args.weights).astype(np.float64)
    signs = mbon_signs(pd.read_parquet(args.neurons), pd.read_csv(args.valence, index_col=0))
    cells, outputs = real.shape
    print(f"KC -> MBON {cells} x {outputs}, {np.count_nonzero(real)} nonzero")
    print(
        f"MBON valence: {(signs > 0.33).sum()} reward, {(signs < -0.33).sum()} punishment,"
        f" {((signs >= -0.33) & (signs <= 0.33)).sum()} mixed\n"
    )

    generator = np.random.default_rng(args.seed)
    projection = generator.normal(size=(args.dimensions, cells)) / np.sqrt(args.dimensions)

    scores: dict[str, list[float]] = {"real": [], "rewired": [], "random": []}
    for repeat in range(args.repeats):
        draw = np.random.default_rng(args.seed + repeat)
        codes = sparse_code(draw.normal(size=(args.items, args.dimensions)), projection, args.active)
        labels = draw.choice([-1.0, 1.0], size=args.items)
        wirings = {
            "real": real,
            "rewired": rewire_preserving_degree(real, draw),
            "random": draw.permutation(real.ravel()).reshape(real.shape),
        }
        for name, mask in wirings.items():
            stored = store_signed(mask, codes, labels, signs)
            scores[name].append(recall_sign(stored, codes, labels, signs))

    print(f"{'wiring':>10} {'signed recall':>14} {'95% interval':>20}")
    summary = {}
    for name, values in scores.items():
        array = np.array(values)
        summary[name] = array
        maker = np.random.default_rng(7)
        draws = [array[maker.integers(0, len(array), len(array))].mean() for _ in range(3000)]
        print(
            f"{name:>10} {array.mean():14.4f}"
            f"  [{np.percentile(draws, 2.5):.4f}, {np.percentile(draws, 97.5):.4f}]"
        )
    print(f"{'chance':>10} {0.5:14.4f}")

    print(f"\n{'comparison':>28} {'difference':>11} {'95% interval':>22}")
    for treatment, control in (("real", "rewired"), ("real", "random"), ("rewired", "random")):
        gap = summary[treatment] - summary[control]
        maker = np.random.default_rng(7)
        draws = [gap[maker.integers(0, len(gap), len(gap))].mean() for _ in range(3000)]
        low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        mark = "resolved" if low > 0 or high < 0 else "NOT resolved"
        label = f"{treatment} over {control}"
        print(f"{label:>28} {gap.mean():+11.4f}  [{low:+.4f}, {high:+.4f}] {mark}")


def demo() -> None:
    """A compartment may only learn the valence its dopamine carries."""
    # Two cells, two outputs: one reward compartment, one punishment.
    mask = np.ones((2, 2))
    signs = np.array([1.0, -1.0])
    codes = np.array([[1.0, 0.0]])
    stored = store_signed(mask, codes, np.array([1.0]), signs)
    # The reward column moves one way and the punishment column the other, so the
    # valence-weighted vote can tell them apart.
    assert stored[0, 0] > 0 and stored[0, 1] < 0, stored
    assert recall_sign(stored, codes, np.array([1.0]), signs) == 1.0
    # A cell that took no part must be untouched.
    assert np.allclose(stored[1], 0.0)
    print("demo ok")


if __name__ == "__main__":
    main()
