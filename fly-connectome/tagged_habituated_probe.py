"""Combine phase 20's tagging/capture, phase 25's habituated dual-store,
and the phase 23 lesson about using absolute (not %-of-own-baseline)
erosion as the fair metric -- the fullest mechanism in this line so far.

Phase 28 (see README.md). fast store: habituated depression (phase 25) on
every event, and a decaying tag. slow store: changes ONLY via capture, and
the capture itself is also habituated against the slow store's own
freshness. Same question as phase 23: does an early salient event protect
an item from being eroded by later unrelated writes -- now measured the
correct (absolute) way from the start.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fetch_mushroom_body import cell_class
from generalization_probe import kc_code_for
from map_text_to_pn import load_pn_positions
from memory import kc_mbon_weights, mbon_drive
from tagging_rescue_probe import DOCUMENT_POOL

FAST_ETA = 0.5
SLOW_ETA = 0.03
CAPTURE_RATE = 1.0
TAG_DECAY = 0.6


def write_event(fast: np.ndarray, tag: np.ndarray, initial: np.ndarray, kc_code: np.ndarray) -> None:
    active = kc_code > 0
    if not active.any():
        tag *= TAG_DECAY
        return
    freshness = fast[active, :].sum(axis=1) / initial[active, :].sum(axis=1)
    fast[active, :] *= (1.0 - FAST_ETA * freshness)[:, None]
    tag *= TAG_DECAY
    tag[active] = 1.0


def capture_event(slow: np.ndarray, tag: np.ndarray, initial: np.ndarray) -> None:
    freshness = slow.sum(axis=1) / initial.sum(axis=1)  # per-KC, over all KCs (tag may be nonzero anywhere)
    slow *= (1.0 - CAPTURE_RATE * tag * freshness)[:, None]


def recall(fast: np.ndarray, slow: np.ndarray, kc_code: np.ndarray) -> float:
    return float(mbon_drive(fast, kc_code).sum() + mbon_drive(slow, kc_code).sum())


def main() -> None:
    data_dir = Path(__file__).parent / "data"
    neurons_df = pd.read_parquet(data_dir / "neurons.parquet")
    matrix = sp.load_npz(data_dir / "adjacency.npz")
    projection = np.load(data_dir / "pn_projection.npy")
    pn_positions, _ = load_pn_positions(data_dir / "neurons.parquet")
    n_total = len(neurons_df)
    kc_mask = (neurons_df["type"].map(cell_class) == "KC").to_numpy()
    mbon_positions = np.flatnonzero((neurons_df["type"].map(cell_class) == "MBON").to_numpy())
    kc_positions = np.flatnonzero(kc_mask)

    codes = [kc_code_for(s, projection, matrix, pn_positions, n_total, kc_mask, 0.10) for s in DOCUMENT_POOL]
    initial = kc_mbon_weights(matrix, kc_positions, mbon_positions)

    n_fillers = 10
    for label, capture in (("captured", True), ("uncaptured", False)):
        rng = np.random.default_rng(0)
        post_encoding_values, post_filler_values = [], []
        for _ in range(50):
            chosen = rng.choice(len(codes), size=n_fillers + 2, replace=False)
            target = codes[chosen[0]]
            salient_event = codes[chosen[1]]
            fillers = [codes[i] for i in chosen[2:]]

            fast, slow, tag = initial.copy(), initial.copy(), np.zeros(initial.shape[0])
            write_event(fast, tag, initial, target)
            if capture:
                write_event(fast, tag, initial, salient_event)
                capture_event(slow, tag, initial)
            post_encoding_values.append(recall(fast, slow, target))

            for filler in fillers:
                write_event(fast, tag, initial, filler)
            post_filler_values.append(recall(fast, slow, target))

        absolute_erosion = np.mean(post_encoding_values) - np.mean(post_filler_values)
        print(f"{label}: post-encoding={np.mean(post_encoding_values):.2f}, "
              f"post-fillers={np.mean(post_filler_values):.2f}, absolute erosion={absolute_erosion:.2f}")


if __name__ == "__main__":
    main()
