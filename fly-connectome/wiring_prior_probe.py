"""Does the real KC-to-MBON wiring store more than a rewired one?

FlyGM (arXiv 2602.17997) instantiated the whole fly connectome as a graph
controller and beat three baselines on locomotion, the sharpest being a
degree-preserving rewiring: every neuron keeps its exact in- and out-degree and
only its partners change. Losing to that means the advantage is *who* is wired
to whom, not how much wiring there is. The paper never tests memory — it
searches for "mushroom body" and finds nothing, because walking does not need
one, even though those neurons sit in the graph it uses.

This asks the question the paper left: the mushroom body is the memory organ, so
does its measured wiring hold associations better than a rewiring that preserves
every degree? The store is the same delta rule everywhere; only the KC-to-MBON
matrix that reads it out differs.

Falsification, fixed before running: if the real wiring does not beat the
degree-preserving rewiring, the connectome carries no memory-specific prior here
and its structure is worth no more than its degree sequence.
"""
from __future__ import annotations

import argparse

import numpy as np


def rewire_preserving_degree(
    weights: np.ndarray, generator: np.random.Generator, rounds: int = 40
) -> np.ndarray:
    """Shuffle partners while holding every row and column degree exactly fixed.

    The curveball trade: take two rows, look at the columns present in exactly
    one of them, and redeal those columns between the two rows keeping each
    row's count. Both degree sequences survive every trade, so the only thing
    that changes is who is wired to whom.

    A first version permuted column indices globally and summed collisions,
    which silently dropped 5,255 of 30,543 synapses — seventeen percent — and
    broke row degree outright. "The real wiring beats a rewiring" then only
    meant it beat a sparser matrix.
    """
    support = [set(np.flatnonzero(row)) for row in weights]
    count = len(support)
    for _ in range(rounds * count):
        first, second = generator.integers(0, count, 2)
        if first == second:
            continue
        shared = support[first] & support[second]
        swappable = list((support[first] | support[second]) - shared)
        if len(swappable) < 2:
            continue
        generator.shuffle(swappable)
        keep = len(support[first]) - len(shared)
        support[first] = shared | set(swappable[:keep])
        support[second] = shared | set(swappable[keep:])

    rewired = np.zeros_like(weights)
    # The same synapse strengths, redealt onto the new support, so the value
    # multiset and the edge count are both unchanged.
    values = generator.permutation(weights[weights != 0])
    cursor = 0
    for row, columns in enumerate(support):
        for column in sorted(columns):
            rewired[row, column] = values[cursor]
            cursor += 1
    return rewired


def sparse_code(patterns: np.ndarray, projection: np.ndarray, active: int) -> np.ndarray:
    """Expand into Kenyon cells and keep the strongest, renormalised."""
    expanded = np.maximum(0.0, patterns @ projection)
    cut = np.partition(expanded, -active, axis=-1)[..., -active][..., None]
    kept = np.where(expanded >= cut, expanded, 0.0)
    norm = np.linalg.norm(kept, axis=-1, keepdims=True)
    return np.divide(kept, norm, out=np.zeros_like(kept), where=norm > 0)


def store(mask: np.ndarray, codes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Write the associations into the synapses the wiring actually provides.

    In the fly the KC-to-MBON synapses are the plastic ones — dopamine changes
    their strength — so the connectome is not the memory, it is the set of places
    a memory may be put. Learning is a delta rule restricted to that support,
    which is what makes one wiring able to hold associations another cannot.

    An earlier version of this file skipped this step entirely and pushed codes
    through the fixed matrix, so every wiring scored exactly chance.
    """
    support = mask != 0
    weights = np.zeros_like(mask, dtype=np.float64)
    for code, value in zip(codes, values):
        predicted = code @ weights
        weights += np.outer(code, value - predicted) * support
    return weights


def recall(weights: np.ndarray, codes: np.ndarray, values: np.ndarray) -> float:
    """Share of stored items whose own value scores highest when cued."""
    read = codes @ weights
    read /= np.linalg.norm(read, axis=1, keepdims=True).clip(min=1e-9)
    targets = values / np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-9)
    return float(((read @ targets.T).argmax(axis=1) == np.arange(len(codes))).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="data/kc_mbon_weights.npy")
    parser.add_argument("--items", type=int, default=60, help="associations stored")
    parser.add_argument("--dimensions", type=int, default=128, help="input pattern width")
    parser.add_argument("--active", type=int, default=40, help="Kenyon cells left on")
    parser.add_argument("--repeats", type=int, default=40, help="rewirings and draws")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    real = np.load(args.weights).astype(np.float64)
    cells, outputs = real.shape
    print(f"KC -> MBON {cells} x {outputs}, {np.count_nonzero(real)} nonzero\n")

    generator = np.random.default_rng(args.seed)
    projection = generator.normal(size=(args.dimensions, cells)) / np.sqrt(args.dimensions)

    scores: dict[str, list[float]] = {"real": [], "rewired": [], "random": []}
    for repeat in range(args.repeats):
        draw = np.random.default_rng(args.seed + repeat)
        patterns = draw.normal(size=(args.items, args.dimensions))
        codes = sparse_code(patterns, projection, args.active)
        # Values live in MBON space and are what the readout has to reproduce.
        values = draw.normal(size=(args.items, outputs))

        readouts = {
            "real": real,
            "rewired": rewire_preserving_degree(real, draw),
            # Same density and same magnitude distribution, no structure at all.
            "random": draw.permutation(real.ravel()).reshape(real.shape),
        }
        for name, mask in readouts.items():
            scores[name].append(recall(store(mask, codes, values), codes, values))

    print(f"{'wiring':>10} {'recall':>8} {'95% interval':>20}")
    summary = {}
    for name, values_ in scores.items():
        array = np.array(values_)
        maker = np.random.default_rng(7)
        draws = [array[maker.integers(0, len(array), len(array))].mean() for _ in range(3000)]
        summary[name] = array
        print(
            f"{name:>10} {array.mean():8.4f}"
            f"  [{np.percentile(draws, 2.5):.4f}, {np.percentile(draws, 97.5):.4f}]"
        )
    print(f"{'chance':>10} {1 / args.items:8.4f}")

    print(f"\n{'comparison':>28} {'difference':>11} {'95% interval':>22}")
    for treatment, control in (("real", "rewired"), ("real", "random"), ("rewired", "random")):
        gap = summary[treatment] - summary[control]
        maker = np.random.default_rng(7)
        draws = [gap[maker.integers(0, len(gap), len(gap))].mean() for _ in range(3000)]
        low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        mark = "resolved" if low > 0 or high < 0 else "NOT resolved"
        print(f"{treatment + ' over ' + control:>28} {gap.mean():+11.4f}  [{low:+.4f}, {high:+.4f}] {mark}")


def demo() -> None:
    """Rewiring must move partners without changing how much wiring there is."""
    weights = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]])
    rewired = rewire_preserving_degree(weights, np.random.default_rng(0))
    assert np.count_nonzero(rewired) == np.count_nonzero(weights), "no synapse may be lost"
    assert np.array_equal((rewired != 0).sum(1), (weights != 0).sum(1)), "row degree must survive"
    assert np.array_equal((rewired != 0).sum(0), (weights != 0).sum(0)), "column degree must survive"
    # Sparse codes must actually be sparse, or the readout is being handed the
    # dense pattern and the wiring is not what is being tested.
    codes = sparse_code(np.random.default_rng(1).normal(size=(4, 8)), np.eye(8), active=3)
    assert (np.count_nonzero(codes, axis=1) == 3).all()
    # Learning may only touch synapses the wiring provides, or the mask is not
    # the thing under test and every wiring scores the same.
    mask = np.array([[1.0, 0.0], [0.0, 1.0]])
    learned = store(mask, np.eye(2), np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert learned[0, 1] == 0.0 and learned[1, 0] == 0.0, learned
    print("demo ok")


if __name__ == "__main__":
    main()
