"""Propagate PN activations (from map_text_to_pn.py) one hop through the real
connectome to the KC layer, then sparsify like the mushroom body does.

Phase 3 of this experiment line (see README.md): first real downstream
dynamics. Still no "stimulus point" definition -- just a KC code per
sentence, which a later step compares across sentences.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fetch_mushroom_body import cell_class

DEFAULT_KEEP_RATIO = 0.10  # Lin et al. 2014: ~5-10% of KCs active at once.


def class_mask(neurons_df: pd.DataFrame, cls: str) -> np.ndarray:
    return (neurons_df["type"].map(cell_class) == cls).to_numpy()


def scatter(values: np.ndarray, positions: np.ndarray, n_total: int) -> np.ndarray:
    """Place per-PN activation values into a full-length vector, zero elsewhere."""
    full = np.zeros(n_total)
    full[positions] = values
    return full


def propagate(stimulus: np.ndarray, matrix: sp.csr_matrix) -> np.ndarray:
    """One synaptic hop: drive received by neuron j = sum_i stimulus[i] * weight(i, j)."""
    return matrix.T.dot(stimulus)


def random_rewire(matrix: sp.csr_matrix, seed: int = 0) -> sp.csr_matrix:
    """Same edge count and same weight values as `matrix`, but every edge's
    target is a uniformly random neuron -- a null model that keeps overall
    density and weight scale identical while destroying the specific real
    wiring, for testing how much of a result is topology versus generic
    sparse-random-projection math."""
    coo = matrix.tocoo()
    rng = np.random.default_rng(seed)
    random_cols = rng.integers(0, matrix.shape[1], size=coo.nnz)
    return sp.coo_matrix((coo.data, (coo.row, random_cols)), shape=matrix.shape).tocsr()


def sparse_code(drive: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Keep only the top keep_ratio fraction of (already non-negative) drive,
    zeroing the rest -- the APL inhibition loop's effect, without modelling APL
    itself."""
    n_active = max(1, int(round(len(drive) * keep_ratio)))
    if n_active >= len(drive):
        return drive
    threshold_idx = np.argpartition(drive, -n_active)[-n_active:]
    coded = np.zeros_like(drive)
    coded[threshold_idx] = drive[threshold_idx]
    return coded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=Path(__file__).parent / "data", type=Path)
    parser.add_argument("--keep-ratio", type=float, default=DEFAULT_KEEP_RATIO)
    args = parser.parse_args()

    neurons_df = pd.read_parquet(args.data_dir / "neurons.parquet")
    matrix = sp.load_npz(args.data_dir / "adjacency.npz")
    pn_data = np.load(args.data_dir / "pn_activations.npz")

    kc_mask = class_mask(neurons_df, "KC")
    n_total = len(neurons_df)
    n_kc = int(kc_mask.sum())

    kc_codes = np.zeros((len(pn_data["sentences"]), n_kc))
    for row, pn_activation in enumerate(pn_data["activations"]):
        stimulus = scatter(pn_activation, pn_data["pn_positions"], n_total)
        drive = np.maximum(0.0, propagate(stimulus, matrix))
        kc_codes[row] = sparse_code(drive[kc_mask], args.keep_ratio)

    out_path = args.data_dir / "kc_codes.npz"
    np.savez(out_path, kc_codes=kc_codes, sentences=pn_data["sentences"])

    print(f"{len(pn_data['sentences'])} sentences -> {n_kc} KC neurons (keep_ratio={args.keep_ratio})")
    for sentence, code in zip(pn_data["sentences"], kc_codes):
        active = int((code > 0).sum())
        preview = str(sentence) if len(str(sentence)) <= 60 else str(sentence)[:57] + "..."
        print(f"  [{active:4d}/{n_kc} active, mean_drive={code.mean():.4f}] {preview}")


if __name__ == "__main__":
    main()
