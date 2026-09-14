"""Opponent-channel valence: does a signed reinforcement event bias the
MBON readout in the expected direction?

Phase 34 (see README.md). RESEARCH.md's own survey: MBON output is opponent
and summed, and "the plasticity is depression: the reinforcement turns down
the output that signals the OPPOSITE valence" (Owald et al. 2015; Aso et
al. 2014) -- not potentiating the agreeing channel, depressing the
disagreeing one. The readout is the DIFFERENCE across the two channels
(Bennett, Philippides & Nowotny 2021's two-channel collapse).

Honesty check up front: this project has no verified mapping from our
fetched MBON type names (MBON14, MBON06, etc.) to which specific
compartments are actually appetitive vs. aversive (Aso et al. 2014 assigns
that per real compartment, but this project hasn't checked our numeric
labels against it). So this splits MBON into two halves *arbitrarily* --
testing whether the opponent-channel MECHANISM behaves correctly given a
sign, not claiming these are the real valence-tagged output neurons.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fetch_mushroom_body import cell_class
from generalization_probe import kc_code_for
from map_text_to_pn import load_pn_positions

ETA = 0.3


def write_signed(weights: np.ndarray, kc_code: np.ndarray, positive_mask: np.ndarray, sign: int, eta: float = ETA) -> None:
    """Depress the columns signaling the OPPOSITE valence to `sign`, for the
    currently active KC rows -- never potentiate the agreeing channel."""
    active = kc_code > 0
    target_cols = ~positive_mask if sign > 0 else positive_mask
    weights[np.ix_(active, target_cols)] *= 1.0 - eta


def valence_score(weights: np.ndarray, kc_code: np.ndarray, positive_mask: np.ndarray) -> float:
    """Positive channel drive minus negative channel drive -- the opponent
    readout, not a raw sum."""
    drive = kc_code @ weights
    return float(drive[positive_mask].sum() - drive[~positive_mask].sum())


def main() -> None:
    data_dir = Path(__file__).parent / "data"
    neurons_df = pd.read_parquet(data_dir / "neurons.parquet")
    matrix = sp.load_npz(data_dir / "adjacency.npz")
    projection = np.load(data_dir / "pn_projection.npy")
    pn_positions, _ = load_pn_positions(data_dir / "neurons.parquet")
    n_total = len(neurons_df)
    kc_mask = (neurons_df["type"].map(cell_class) == "KC").to_numpy()
    kc_positions = np.flatnonzero(kc_mask)
    mbon_positions = np.flatnonzero((neurons_df["type"].map(cell_class) == "MBON").to_numpy())

    template = np.asarray(matrix[kc_positions][:, mbon_positions].todense(), dtype=float)
    n_mbon = len(mbon_positions)
    positive_mask = np.zeros(n_mbon, dtype=bool)
    positive_mask[: n_mbon // 2] = True  # arbitrary split, see module docstring

    sentence = "A cup of coffee sat cold on the desk."
    code = kc_code_for(sentence, projection, matrix, pn_positions, n_total, kc_mask, 0.10)

    baseline = valence_score(template, code, positive_mask)
    print(f"baseline valence score (before any reinforcement): {baseline:+.2f}\n")

    for sign, label in ((+1, "positive/appetitive"), (-1, "negative/aversive")):
        weights = template.copy()
        write_signed(weights, code, positive_mask, sign)
        after = valence_score(weights, code, positive_mask)
        print(f"after {label} reinforcement: {after:+.2f}  (shift from baseline: {after - baseline:+.2f})")


if __name__ == "__main__":
    main()
