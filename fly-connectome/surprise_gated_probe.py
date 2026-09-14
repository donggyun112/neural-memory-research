"""Does habituation (write strength scaled by how fresh/undepressed the
active KCs still are, memory.write_surprise_gated) resolve phase 7's
capacity problem on its own, the way pushing fast_eta near 1 did in
phase 22 -- without needing a second store or a competitive budget?

Phase 24 (see README.md). Runs the same phase-7 battery (write everything
in a 4-topic, 12-sentence pool in sequence, track load and same/diff-topic
interference) under write_surprise_gated instead of the flat write().
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
from memory import kc_mbon_weights, mbon_drive, write_surprise_gated


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
    weights = initial.copy()

    # exact-repeat recognition
    first_sentence = order[0][1]
    before = float(mbon_drive(weights, codes[first_sentence]).sum())
    write_surprise_gated(weights, initial, codes[first_sentence])
    after = float(mbon_drive(weights, codes[first_sentence]).sum())
    print(f"exact repeat: {before:.2f} -> {after:.2f}  ({100 * (after - before) / before:+.1f}%)\n")

    # capacity battery
    weights = initial.copy()
    history = {s: [float(mbon_drive(weights, codes[s]).sum())] for _, s in order}
    for _, w_sentence in order:
        write_surprise_gated(weights, initial, codes[w_sentence])
        for _, s in order:
            history[s].append(float(mbon_drive(weights, codes[s]).sum()))

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

    # habituation check: repeat the SAME sentence many times, watch the drop shrink each time
    print("\nhabituation on repeated exposure to the same sentence:")
    weights = initial.copy()
    sentence = order[0][1]
    prev = float(mbon_drive(weights, codes[sentence]).sum())
    for i in range(5):
        write_surprise_gated(weights, initial, codes[sentence])
        current = float(mbon_drive(weights, codes[sentence]).sum())
        print(f"  exposure {i + 1}: {prev:10.2f} -> {current:10.2f}  ({100 * (current - prev) / prev:+.2f}%)")
        prev = current


if __name__ == "__main__":
    main()
