"""Is this circuit's memory a Bloom filter over KC identities?

Phase 36 found the winning write rule is saturating depression: `eta` near 1
nearly zeroes every KC row that was active, so what the synapses hold is close
to a binary mark on "this KC has been part of something written". If that is
really what the memory is, its capacity is not an empirical curve to be
measured mechanism by mechanism -- it is the false-positive rate of a Bloom
filter, and it can be predicted before running anything.

An unwritten document reads as written only if *every* KC it activates was
already marked by some other document. With `m` Kenyon cells, `k` of them
active per document and `N` documents written,

    p = 1 - (1 - k/m)^N          each KC marked by at least one write
    AUC = 1 - 0.5 * p^k          a false positive is a tie, which scores 0.5

Nothing here is fitted. `k` and `m` are counted from the circuit and `N` is the
load. The prediction is falsifiable in both directions: if the measured curve
tracks it, the capacity of this memory follows from two integers; if it sits
below, the KC codes are not behaving like independent random sets, and the
shuffled-code control says whether that is the reason.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fetch_mushroom_body import cell_class
from map_text_to_pn import load_pn_positions
from memory import kc_mbon_weights
from recognition_under_load_probe import DEFAULT_FEATURES, auc, longmemeval_codes


def predicted_auc(active: float, total: int, written: int) -> float:
    """Bloom false-positive rate turned into the same AUC the probe measures."""
    if not 0 < active <= total:
        raise ValueError("active cells must lie in (0, total]")
    marked = 1.0 - (1.0 - active / total) ** written
    return 1.0 - 0.5 * marked**active


def saturating_scores(
    initial: np.ndarray, codes: np.ndarray, written: list[int], eta: float, binary: bool = False
) -> np.ndarray:
    """Write the chosen documents with near-total depression, then read all.

    The default multiplies, so a row written twice lands at (1-eta) squared and
    the state counts how many times each cell took part. Setting ``binary``
    clamps instead, which is the filter the prediction above actually describes:
    marked or not, with no memory of multiplicity.
    """
    weights = initial.copy()
    for index in written:
        active = codes[index] > 0
        if binary:
            weights[active, :] = initial[active, :] * (1.0 - eta)
        else:
            weights[active, :] *= 1.0 - eta
    pristine = codes @ initial.sum(axis=1)
    after = codes @ weights.sum(axis=1)
    return -(after - pristine) / pristine


def shuffled_codes(codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same number of active cells per document, but chosen independently.

    The prediction assumes each document marks a random subset. Real codes come
    from a shared projection and a shared corpus, so they overlap more than that.
    This control keeps every count identical and removes only the correlation.
    """
    out = np.zeros_like(codes)
    for row in range(len(codes)):
        active = np.flatnonzero(codes[row] > 0)
        picked = rng.choice(codes.shape[1], size=len(active), replace=False)
        out[row, picked] = codes[row, active]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=Path(__file__).parent / "data", type=Path)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--keep-ratios", default="0.02,0.05,0.10,0.20,0.40")
    parser.add_argument("--loads", default="8,32,128,512")
    parser.add_argument("--eta", type=float, default=0.99)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    neurons_df = pd.read_parquet(args.data_dir / "neurons.parquet")
    matrix = sp.load_npz(args.data_dir / "adjacency.npz")
    pn_positions, _ = load_pn_positions(args.data_dir / "neurons.parquet")
    n_total = len(neurons_df)
    kc_mask = (neurons_df["type"].map(cell_class) == "KC").to_numpy()
    kc_positions = np.flatnonzero(kc_mask)
    mbon_positions = np.flatnonzero((neurons_df["type"].map(cell_class) == "MBON").to_numpy())
    initial = kc_mbon_weights(matrix, kc_positions, mbon_positions)
    total_kc = len(kc_positions)

    loads = [int(value) for value in args.loads.split(",")]
    ratios = [float(value) for value in args.keep_ratios.split(",")]
    documents = 2 * max(loads)
    print(f"{total_kc} Kenyon cells, eta={args.eta}, {args.trials} trials per cell")
    print(
        f"{'keep':>6} {'active k':>9} {'written':>8} {'predicted':>10}"
        f" {'real, counting':>18} {'real, binary':>18} {'shuffled, binary':>18}"
    )
    rng = np.random.default_rng(args.seed)
    for ratio in ratios:
        codes = np.array(
            longmemeval_codes(
                args.features,
                documents,
                args.data_dir,
                matrix,
                pn_positions,
                n_total,
                kc_mask,
                ratio,
                args.seed,
            )
        )
        active = float((codes > 0).sum(axis=1).mean())
        shuffled = shuffled_codes(codes, rng)
        for load in loads:
            if 2 * load > len(codes):
                continue
            counting, binary, fake = [], [], []
            for _ in range(args.trials):
                chosen = rng.choice(len(codes), size=2 * load, replace=False)
                first, second = sorted(chosen[:load].tolist()), sorted(chosen[load:].tolist())
                for source, sink, is_binary in (
                    (codes, counting, False),
                    (codes, binary, True),
                    (shuffled, fake, True),
                ):
                    scores = saturating_scores(initial, source, first, args.eta, is_binary)
                    sink.append(auc(scores[first], scores[second]))
            cells = ' '.join(
                f"{np.mean(block):12.4f}±{np.std(block):.3f}"
                for block in (counting, binary, fake)
            )
            print(
                f"{ratio:>6.2f} {active:9.1f} {load:>8}"
                f" {predicted_auc(active, total_kc, load):10.4f} {cells}"
            )


if __name__ == "__main__":
    main()
