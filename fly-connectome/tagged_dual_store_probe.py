"""Combine phase 20 (tagging/capture) with phase 18/22 (dual fast/slow
store) into one mechanism, instead of leaving them as separate probes.

Phase 23 (see README.md). Closer to the actual biological claim behind
synaptic tagging and capture: a fast/labile store changes on every event and
leaves a decaying tag; the slow/durable store does NOT change on ordinary
events at all -- it only changes when a later salient event "captures"
whatever is still tagged, content-free (the capturing event's own KCs don't
need to overlap with what gets captured). This tests something phase 18 and
20 couldn't separately: does an early salient event determine whether an
item becomes a LASTING memory (protected in the slow store) or stays
transient (only in the fast store, and so still vulnerable to being
overwritten by later unrelated writes)?
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
CAPTURE_RATE = 1.0
TAG_DECAY = 0.6


def write_event(fast: np.ndarray, tag: np.ndarray, kc_code: np.ndarray) -> None:
    """Every event updates only the fast store, and refreshes the tag on
    whatever KCs it touched."""
    active = kc_code > 0
    fast[active, :] *= 1.0 - FAST_ETA
    tag *= TAG_DECAY
    tag[active] = 1.0


def capture_event(slow: np.ndarray, tag: np.ndarray) -> None:
    """A salient event consolidates whatever is currently tagged into the
    slow store -- content-free, regardless of which KCs this event itself
    touched."""
    slow *= (1.0 - CAPTURE_RATE * tag)[:, None]


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
    template = kc_mbon_weights(matrix, kc_positions, mbon_positions)

    n_fillers = 10  # unrelated writes that happen after the target, regardless of capture

    for label, capture in (("captured (salient event follows immediately)", True), ("uncaptured (no early salient event)", False)):
        rng = np.random.default_rng(0)  # same episodes in both conditions, for a fair comparison
        erosion_pct = []
        post_encoding_values: list[float] = []
        post_filler_values: list[float] = []
        for _ in range(50):
            chosen = rng.choice(len(codes), size=n_fillers + 2, replace=False)
            target = codes[chosen[0]]
            salient_event = codes[chosen[1]]
            fillers = [codes[i] for i in chosen[2:]]

            fast, slow, tag = template.copy(), template.copy(), np.zeros(template.shape[0])
            write_event(fast, tag, target)
            if capture:
                write_event(fast, tag, salient_event)
                capture_event(slow, tag)
            post_encoding = recall(fast, slow, target)  # right after target is (possibly) consolidated

            for filler in fillers:
                write_event(fast, tag, filler)
            post_fillers = recall(fast, slow, target)  # after unrelated load

            erosion_pct.append(100 * (post_fillers - post_encoding) / post_encoding)
            post_encoding_values.append(post_encoding)
            post_filler_values.append(post_fillers)

        absolute_erosion = np.mean(post_encoding_values) - np.mean(post_filler_values)
        print(f"{label}:")
        print(f"  post-encoding avg: {np.mean(post_encoding_values):10.2f}")
        print(f"  post-fillers avg:  {np.mean(post_filler_values):10.2f}")
        print(f"  absolute erosion:  {absolute_erosion:10.2f}  "
              f"(the fair comparison -- capture's own extra one-time depression makes its "
              f"post-encoding baseline smaller, so % relative to that baseline is misleading here)")
        print(f"  erosion as % of its own baseline: {np.mean(erosion_pct):+.2f}% (n=50 episodes)")


if __name__ == "__main__":
    main()
