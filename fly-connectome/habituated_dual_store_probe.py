"""Combine phase 24's habituation (write strength scaled by remaining
freshness) with phase 18's dual fast/slow store -- each store habituates
against its own pristine reference, independently.

Phase 25 (see README.md). Runs the same battery as phase 18/22/24 for a
direct three-way comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from capacity_probe import TOPICS
from fetch_mushroom_body import cell_class
from generalization_probe import kc_code_for
from map_text_to_pn import load_pn_positions
from memory import kc_mbon_weights, mbon_drive

FAST_ETA = 0.5
SLOW_ETA = 0.03


def habituated_dual_write(fast: np.ndarray, slow: np.ndarray, initial: np.ndarray, kc_code: np.ndarray) -> None:
    active = kc_code > 0
    if not active.any():
        return
    fast_freshness = fast[active, :].sum(axis=1) / initial[active, :].sum(axis=1)
    slow_freshness = slow[active, :].sum(axis=1) / initial[active, :].sum(axis=1)
    fast[active, :] *= (1.0 - FAST_ETA * fast_freshness)[:, None]
    slow[active, :] *= (1.0 - SLOW_ETA * slow_freshness)[:, None]


def dual_drive(fast: np.ndarray, slow: np.ndarray, kc_code: np.ndarray) -> float:
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

    order = [(topic, s) for topic, sents in TOPICS.items() for s in sents]
    codes = {s: kc_code_for(s, projection, matrix, pn_positions, n_total, kc_mask, 0.10) for _, s in order}
    topic_of = {s: t for t, s in order}
    initial = kc_mbon_weights(matrix, kc_positions, mbon_positions)

    first_sentence = order[0][1]
    fast, slow = initial.copy(), initial.copy()
    before = dual_drive(fast, slow, codes[first_sentence])
    habituated_dual_write(fast, slow, initial, codes[first_sentence])
    after = dual_drive(fast, slow, codes[first_sentence])
    print(f"exact repeat: {before:.2f} -> {after:.2f}  ({100 * (after - before) / before:+.1f}%)\n")

    fast, slow = initial.copy(), initial.copy()
    history = {s: [dual_drive(fast, slow, codes[s])] for _, s in order}
    for _, w_sentence in order:
        habituated_dual_write(fast, slow, initial, codes[w_sentence])
        for _, s in order:
            history[s].append(dual_drive(fast, slow, codes[s]))

    same_topic_pct, diff_topic_pct = [], []
    for i, (_, w_sentence) in enumerate(order):
        for _, s in order:
            if s == w_sentence:
                continue
            b, a = history[s][i], history[s][i + 1]
            pct = 100 * (a - b) / b
            (same_topic_pct if topic_of[s] == topic_of[w_sentence] else diff_topic_pct).append(pct)

    print("one-shot interference on OTHER sentences:")
    print(f"  same topic:      {np.mean(same_topic_pct):+.2f}%")
    print(f"  different topic: {np.mean(diff_topic_pct):+.2f}%")

    trace = history[first_sentence]
    print(f"\nload curve for the first item written:")
    print(f"  after 0 intervening writes (its own):  {100 * (trace[1] - trace[0]) / trace[0]:+.1f}%")
    for k in (3, 6, 11):
        print(f"  after {k:2d} intervening writes:          {100 * (trace[k + 1] - trace[0]) / trace[0]:+.1f}%")


if __name__ == "__main__":
    main()
