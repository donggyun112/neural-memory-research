"""Write a batch of sentences across several topics, in sequence, and track
every sentence's recall after every single write.

Phase 7 probe (see README.md). Answers two things the single-pair tests in
memory.py / generalization_probe.py couldn't: (1) does interference from one
write stay concentrated on same-topic items more than different-topic ones,
once there's more than one topic in play, and (2) how fast does an early
item's recall drift as unrelated writes pile up (load).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fetch_mushroom_body import cell_class
from generalization_probe import kc_code_for
from kc_response import class_mask, random_rewire
from map_text_to_pn import load_pn_positions
from memory import WRITE_MODES, kc_mbon_weights, mbon_drive

TOPICS = {
    "fire": [
        "The fire alarm rang loudly in the empty hallway.",
        "A loud alarm blared through the deserted corridor.",
        "Sirens wailed through the abandoned building at night.",
    ],
    "drink": [
        "A cup of coffee sat cold on the desk.",
        "Her mug of tea grew cold on the table.",
        "An untouched glass of water rested on the counter.",
    ],
    "weather": [
        "Rain tapped steadily against the window all afternoon.",
        "Drizzle fell softly on the glass throughout the day.",
        "Snow drifted quietly past the frosted pane that evening.",
    ],
    "dog": [
        "The dog chased a ball across the sunny park.",
        "A puppy ran after a toy in the green field.",
        "Children played fetch with their pet near the meadow.",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=Path(__file__).parent / "data", type=Path)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--keep-ratio", type=float, default=0.10)
    parser.add_argument("--write-mode", choices=WRITE_MODES, default="unbudgeted")
    parser.add_argument(
        "--random-control", action="store_true",
        help="replace the real connectome with a degree/weight-matched random rewiring",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for --random-control")
    args = parser.parse_args()
    write_fn = WRITE_MODES[args.write_mode]

    neurons_df = pd.read_parquet(args.data_dir / "neurons.parquet")
    matrix = sp.load_npz(args.data_dir / "adjacency.npz")
    if args.random_control:
        matrix = random_rewire(matrix, seed=args.seed)
    projection = np.load(args.data_dir / "pn_projection.npy")
    pn_positions, _ = load_pn_positions(args.data_dir / "neurons.parquet")
    n_total = len(neurons_df)
    kc_mask = class_mask(neurons_df, "KC")
    kc_positions = np.flatnonzero(kc_mask)
    mbon_positions = np.flatnonzero((neurons_df["type"].map(cell_class) == "MBON").to_numpy())
    weights = kc_mbon_weights(matrix, kc_positions, mbon_positions)

    order = [(topic, s) for topic, sents in TOPICS.items() for s in sents]
    codes = {s: kc_code_for(s, projection, matrix, pn_positions, n_total, kc_mask, args.keep_ratio) for _, s in order}
    topic_of = {s: t for t, s in order}

    # history[s] = recall value right before writing item 0, and right after each write
    history: dict[str, list[float]] = {s: [float(mbon_drive(weights, codes[s]).sum())] for _, s in order}
    for _, w_sentence in order:
        write_fn(weights, codes[w_sentence], eta=args.eta)
        for _, s in order:
            history[s].append(float(mbon_drive(weights, codes[s]).sum()))

    # one-shot interference: effect of writing item i, isolated to step i -> i+1,
    # on every OTHER item j, split same-topic vs different-topic.
    same_topic_pct, diff_topic_pct = [], []
    for i, (_, w_sentence) in enumerate(order):
        for _, s in order:
            if s == w_sentence:
                continue
            before, after = history[s][i], history[s][i + 1]
            pct = 100 * (after - before) / before
            (same_topic_pct if topic_of[s] == topic_of[w_sentence] else diff_topic_pct).append(pct)

    print(f"one-shot interference on OTHER sentences (mean % change in recall):")
    print(f"  same topic:      {np.mean(same_topic_pct):+.2f}%  (n={len(same_topic_pct)})")
    print(f"  different topic: {np.mean(diff_topic_pct):+.2f}%  (n={len(diff_topic_pct)})")

    first_sentence = order[0][1]
    trace = history[first_sentence]
    print(f"\nload curve for the first item written (\"{first_sentence[:40]}...\"):")
    print(f"  after 0 intervening writes (its own):  {trace[1]:.1f}  ({100 * (trace[1] - trace[0]) / trace[0]:+.1f}%)")
    for k in (3, 6, 11):
        pct = 100 * (trace[k + 1] - trace[0]) / trace[0]
        print(f"  after {k:2d} intervening writes:          {trace[k + 1]:.1f}  ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
