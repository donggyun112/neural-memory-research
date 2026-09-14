"""Score each sentence by how much its KC code stands out from the others.

Phase 4 of this experiment line (see README.md): a first, falsifiable
definition of "stimulus point" -- not the definition, a candidate one. KC
codes are sparse identity codes (Lin et al. 2014), so "how different" is
measured as low overlap of the *active KC set* with the rest of the
document, following how the olfactory literature compares sparse codes
(overlap = similarity), not raw-magnitude cosine distance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def active_set(code: np.ndarray) -> frozenset[int]:
    return frozenset(np.flatnonzero(code))


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def stimulus_scores(kc_codes: np.ndarray) -> np.ndarray:
    """1 - mean similarity to every other sentence's KC code. Undefined
    (returned as 0) with fewer than two sentences -- there's nothing to
    stand out from."""
    n = len(kc_codes)
    if n < 2:
        return np.zeros(n)
    sets = [active_set(code) for code in kc_codes]
    scores = np.zeros(n)
    for i in range(n):
        similarities = [jaccard(sets[i], sets[j]) for j in range(n) if j != i]
        scores[i] = 1.0 - float(np.mean(similarities))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=Path(__file__).parent / "data", type=Path)
    args = parser.parse_args()

    data = np.load(args.data_dir / "kc_codes.npz")
    scores = stimulus_scores(data["kc_codes"])

    order = np.argsort(-scores)
    print("Ranked by stimulus score (higher = more distinct KC code among these sentences):")
    for rank, i in enumerate(order, start=1):
        sentence = str(data["sentences"][i])
        preview = sentence if len(sentence) <= 70 else sentence[:67] + "..."
        print(f"  {rank}. [{scores[i]:.3f}] {preview}")


if __name__ == "__main__":
    main()
