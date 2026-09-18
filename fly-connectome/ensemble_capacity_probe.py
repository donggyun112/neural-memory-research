"""Under the fly's own rule, does the 75-vector carry more than the scalar?

board/001 recorded angle 1 as closed: the MBON ensemble is one number with noise
on it, 95.7% of the variance in the first component. That number turned out to
be a property of `teach()` as implemented — a signed push-pull rule that moves
both compartment families in lockstep on every lesson, which is rank-1-dominant
by construction. Under a depression-only rule, where an outcome perturbs only the
compartments it recruits, the same measurement gives 48.6%.

So the claim has been retracted, but retracting is not the same as answering. A
spread spectrum says the dimensions move independently; it does not say anything
readable lives in them. This finishes the retraction by asking the question the
first measurement was supposed to answer.

Falsification, fixed before running: if a linear probe on the 75-vector does not
beat the same probe on the collapsed scalar at recovering which outcome a
stimulus was taught, the extra dimensions are redundant and angle 1 closes again
— this time on the fly's own rule rather than on mine.

This is bookkeeping on a retraction, not a live direction. Phase 80's conclusion
about resemblance-based memory rests on three mechanisms and does not move either
way with this result.
"""
from __future__ import annotations

import argparse

import numpy as np

from mushroom_body import MushroomBody, encode, load


def depress_only(body: MushroomBody, stimulus: np.ndarray, outcome: float) -> None:
    """Dopamine only ever weakens, and only where it arrives.

    The rule in `mushroom_body.teach` moves both compartment families on every
    lesson, one down and one up. This moves only the family the outcome actually
    recruits, and only downward, which is what the fly's plasticity is described
    as doing.
    """
    code = encode(body.circuit, stimulus)
    recruited = np.maximum(outcome * body.circuit.signs, 0.0)
    body.synapses = np.clip(
        body.synapses - body.rate * np.outer(code, recruited) * body.support, 0.0, None
    )


def fit_probe(features: np.ndarray, labels: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Least squares with a ridge term, so a 1-column design is not degenerate."""
    design = np.column_stack([features, np.ones(len(features))])
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ labels)


def accuracy(weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
    design = np.column_stack([features, np.ones(len(features))])
    return float((np.sign(design @ weights) == labels).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taught", type=int, default=120, help="stimuli the organ learns")
    parser.add_argument("--held-out", type=int, default=60)
    parser.add_argument("--inputs", type=int, default=384)
    parser.add_argument("--active", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    circuit = load(inputs=args.inputs, active=args.active, seed=args.seed)
    print(f"{circuit.weights.shape[0]} Kenyon cells -> {circuit.weights.shape[1]} MBONs\n")

    results: dict[str, list[float]] = {name: [] for name in ("ensemble", "scalar", "untaught")}
    spectra: list[float] = []
    for repeat in range(args.repeats):
        draw = np.random.default_rng(args.seed + repeat)
        total = args.taught + args.held_out
        stimuli = draw.normal(size=(total, args.inputs))
        labels = draw.choice([-1.0, 1.0], size=total)

        organ = MushroomBody(circuit)
        bare = MushroomBody(circuit)
        for stimulus, label in zip(stimuli[: args.taught], labels[: args.taught]):
            depress_only(organ, stimulus, label)

        # Read every taught and held-out stimulus both ways: the full ensemble,
        # and the same ensemble collapsed the way sense() collapses it.
        def read(body: MushroomBody) -> np.ndarray:
            return np.array([encode(circuit, s) @ body.synapses for s in stimuli])

        ensemble = read(organ)
        scalar = (ensemble @ -circuit.signs)[:, None]
        untaught = read(bare)

        changes = ensemble[: args.taught] - untaught[: args.taught]
        singular = np.linalg.svd(changes, compute_uv=False)
        spectra.append(float(singular[0] ** 2 / (singular**2).sum()))

        train, test = slice(0, args.taught), slice(args.taught, total)
        for name, features in (
            ("ensemble", ensemble),
            ("scalar", scalar),
            # The untaught organ is the floor: whatever a probe can read off a
            # circuit that learned nothing is the projection talking, not memory.
            ("untaught", untaught),
        ):
            probe = fit_probe(features[train], labels[train])
            results[name].append(accuracy(probe, features[test], labels[test]))

    print(f"first singular component, depression-only: {np.mean(spectra):.1%}")
    print(f"\n{'readout':>12} {'held-out accuracy':>18} {'95% interval':>22}")
    summary = {}
    for name, values in results.items():
        array = np.array(values)
        summary[name] = array
        maker = np.random.default_rng(7)
        draws = [array[maker.integers(0, len(array), len(array))].mean() for _ in range(3000)]
        print(
            f"{name:>12} {array.mean():18.4f}"
            f"  [{np.percentile(draws, 2.5):.4f}, {np.percentile(draws, 97.5):.4f}]"
        )
    print(f"{'chance':>12} {0.5:18.4f}")

    gap = summary["ensemble"] - summary["scalar"]
    maker = np.random.default_rng(7)
    draws = [gap[maker.integers(0, len(gap), len(gap))].mean() for _ in range(3000)]
    low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    print(f"\nensemble over scalar: {gap.mean():+.4f}  [{low:+.4f}, {high:+.4f}]")
    print(
        "VERDICT: "
        + (
            "the 75 carry something the scalar does not"
            if low > 0
            else "redundant — angle 1 closes again, on the fly's rule this time"
        )
    )


if __name__ == "__main__":
    main()
