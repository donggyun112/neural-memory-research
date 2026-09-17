"""Does a later lesson erase an earlier one less in the real wiring?

Phase 38's valence task could not tell the wirings apart, and the fault was
mine: the signs were indexed by MBON column, so rewiring — which only moves
which Kenyon cell reaches which MBON — left the valence map intact and every
condition scored the same. The structure was supplied by the scoring rather than
demanded from the wiring.

Interference has no such escape. It is a property of the matrix alone: two
lessons collide to the extent they land on the same synapses, and nothing about
the readout can hide that. The structure probe measured the collision surface
directly — a real Kenyon cell reaches 7.3 compartments where a degree-preserving
rewiring reaches 9.9 — so the prediction is specific and falsifiable.

Falsification, fixed before running: if retention after an interfering lesson is
no higher in the real wiring than in the rewiring, then reaching fewer
compartments buys no protection, and the anatomy's selectivity does not serve
the separation of lessons.
"""
from __future__ import annotations

import argparse

import numpy as np

from wiring_prior_probe import rewire_preserving_degree, sparse_code


def learn(mask: np.ndarray, codes: np.ndarray, values: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """Delta-rule writing restricted to the synapses the wiring provides."""
    support = mask != 0
    weights = np.zeros_like(mask, dtype=np.float64)
    for code, value in zip(codes, values):
        weights += rate * np.outer(code, value - code @ weights) * support
    return weights


def retention(weights: np.ndarray, codes: np.ndarray, values: np.ndarray) -> float:
    """Share of the cued items whose own value still scores highest."""
    read = codes @ weights
    read /= np.linalg.norm(read, axis=1, keepdims=True).clip(min=1e-9)
    targets = values / np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-9)
    return float(((read @ targets.T).argmax(axis=1) == np.arange(len(codes))).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="data/kc_mbon_weights.npy")
    parser.add_argument("--first", type=int, default=300, help="items in the lesson under test")
    parser.add_argument("--second", type=int, default=300, help="items in the interfering lesson")
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--active", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    real = np.load(args.weights).astype(np.float64)
    cells, outputs = real.shape
    print(f"KC -> MBON {cells} x {outputs}, {np.count_nonzero(real)} nonzero")
    print(f"lesson A: {args.first} items, then lesson B: {args.second} items\n")

    generator = np.random.default_rng(args.seed)
    projection = generator.normal(size=(args.dimensions, cells)) / np.sqrt(args.dimensions)

    scores: dict[str, dict[str, list[float]]] = {
        name: {"alone": [], "after": []} for name in ("real", "rewired", "random")
    }
    for repeat in range(args.repeats):
        draw = np.random.default_rng(args.seed + repeat)
        patterns = draw.normal(size=(args.first + args.second, args.dimensions))
        codes = sparse_code(patterns, projection, args.active)
        values = draw.normal(size=(args.first + args.second, outputs))
        first_codes, first_values = codes[: args.first], values[: args.first]

        wirings = {
            "real": real,
            "rewired": rewire_preserving_degree(real, draw),
            "random": draw.permutation(real.ravel()).reshape(real.shape),
        }
        for name, mask in wirings.items():
            # Learning A alone fixes what this wiring can hold at all, so the
            # retention below is measured against the wiring's own ceiling
            # rather than against the other wirings' ceilings.
            alone = learn(mask, first_codes, first_values)
            both = learn(mask, codes, values)
            scores[name]["alone"].append(retention(alone, first_codes, first_values))
            scores[name]["after"].append(retention(both, first_codes, first_values))

    print(f"{'wiring':>10} {'A alone':>9} {'A after B':>11} {'kept':>8}")
    kept: dict[str, np.ndarray] = {}
    for name, values_ in scores.items():
        alone = np.array(values_["alone"])
        after = np.array(values_["after"])
        # The ratio is what isolates interference: a wiring that stores less to
        # begin with would otherwise look more robust simply by having less to
        # lose.
        kept[name] = after / alone.clip(min=1e-9)
        print(f"{name:>10} {alone.mean():9.4f} {after.mean():11.4f} {kept[name].mean():8.4f}")

    print(f"\n{'comparison':>28} {'difference':>11} {'95% interval':>22}")
    for control in ("rewired", "random"):
        gap = kept["real"] - kept[control]
        maker = np.random.default_rng(7)
        draws = [gap[maker.integers(0, len(gap), len(gap))].mean() for _ in range(3000)]
        low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        mark = "resolved" if low > 0 or high < 0 else "NOT resolved"
        print(f"{'kept: real over ' + control:>28} {gap.mean():+11.4f}  [{low:+.4f}, {high:+.4f}] {mark}")


def demo() -> None:
    """Retention must fall when a second lesson lands on the same synapses."""
    mask = np.ones((6, 3))
    draw = np.random.default_rng(0)
    codes = sparse_code(draw.normal(size=(4, 6)), np.eye(6), active=3)
    values = draw.normal(size=(4, 3))
    alone = retention(learn(mask, codes[:2], values[:2]), codes[:2], values[:2])
    after = retention(learn(mask, codes, values), codes[:2], values[:2])
    assert alone >= after, (alone, after)
    # A wiring with no synapses can hold nothing, and must not read as perfect.
    empty = learn(np.zeros((6, 3)), codes[:2], values[:2])
    assert np.allclose(empty, 0.0)
    print("demo ok")


if __name__ == "__main__":
    main()
