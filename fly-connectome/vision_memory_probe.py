"""Write/recall memory on the vision circuit, across more images per topic
-- does interference stay higher within-topic than across-topic, the same
question capacity_probe.py asked for text (phase 7)?

Phase 14/15 (see README.md). Phase 14's first attempt used raw R1-R6
activation directly as the write/recall code -- real images leave ~100% of
R1-R6 nonzero (unlike text's PN activation, naturally ~50% sparse from ReLU
on random projections), so every write depressed nearly the entire matrix
regardless of image identity: a uniform effect, not memory. Fixed by
mirroring the text pipeline's shape: R1-R6 (dense, PN-equivalent)
propagates one hop to the downstream lamina ("other", KC-equivalent), THEN
gets sparse-coded (top 10%, kc_response.sparse_code) before being used for
write/recall against the real "other -> other" lamina recurrent synapses
(the honest MBON-equivalent available without fetching the medulla).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fetch_optic_lobe import photoreceptor_class
from image_to_photoreceptors import build_stimulus, fetch_image
from kc_response import propagate, sparse_code
from memory import mbon_drive, write

KEEP_RATIO = 0.10
ETA = 0.3

IMAGE_TOPICS = {
    "coffee": {
        "cup": "https://upload.wikimedia.org/wikipedia/commons/4/45/A_small_cup_of_coffee.JPG",
        "beans": "https://upload.wikimedia.org/wikipedia/commons/c/c5/Roasted_coffee_beans.jpg",
    },
    "dog": {
        "shaggy": "https://upload.wikimedia.org/wikipedia/commons/d/d1/Shaggy_Dog_running.jpg",
        "golden": "https://upload.wikimedia.org/wikipedia/commons/4/47/Golden_retriever.jpg",
    },
    "alarm": {
        "bell": "https://upload.wikimedia.org/wikipedia/commons/8/8e/Kobishi_Electric_MSB-63A_fire_alarm_bell.JPG",
        "smoke_detector": "https://upload.wikimedia.org/wikipedia/commons/d/d9/Smoke_detector.JPG",
    },
}


def lamina_code(stimulus: np.ndarray, matrix: sp.csr_matrix, other_mask: np.ndarray, keep_ratio: float) -> np.ndarray:
    drive = np.maximum(0.0, propagate(stimulus, matrix))
    return sparse_code(drive[other_mask], keep_ratio)


def main() -> None:
    data_dir = "data_optic"
    neurons_df = pd.read_parquet(f"{data_dir}/neurons.parquet")
    matrix = sp.load_npz(f"{data_dir}/adjacency.npz")
    retinotopy = dict(np.load(f"{data_dir}/retinotopy.npz"))

    classes = neurons_df["type"].map(photoreceptor_class)
    other_mask = (classes == "other").to_numpy()
    other_positions = np.flatnonzero(other_mask)
    n_other = len(other_positions)

    order = [(topic, name) for topic, images in IMAGE_TOPICS.items() for name in images]
    topic_of = {name: topic for topic, name in order}
    codes = {}
    for topic, name in order:
        url = IMAGE_TOPICS[topic][name]
        image = fetch_image(url)
        stimulus = build_stimulus(image, neurons_df, retinotopy)
        codes[name] = lamina_code(stimulus, matrix, other_mask, KEEP_RATIO)
        print(f"{topic}/{name}: {int((codes[name] > 0).sum())}/{n_other} lamina neurons active")

    active_sets = {name: set(np.flatnonzero(code)) for name, code in codes.items()}
    print("\npairwise active-set overlap (Jaccard), same-topic marked with *:")
    names = [name for _, name in order]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            union = active_sets[a] | active_sets[b]
            jaccard = len(active_sets[a] & active_sets[b]) / len(union) if union else 0.0
            marker = "*" if topic_of[a] == topic_of[b] else " "
            print(f"  {marker} {a} vs {b}: {jaccard:.3f}")

    fresh_weights = np.asarray(matrix[other_positions][:, other_positions].todense(), dtype=float)
    weights = fresh_weights.copy()

    # history[name] = recall right before writing item 0, and after each write
    history = {name: [float(mbon_drive(weights, codes[name]).sum())] for _, name in order}
    for _, w_name in order:
        write(weights, codes[w_name], eta=ETA)
        for _, name in order:
            history[name].append(float(mbon_drive(weights, codes[name]).sum()))

    same_topic_pct, diff_topic_pct = [], []
    for i, (_, w_name) in enumerate(order):
        for _, name in order:
            if name == w_name:
                continue
            before, after = history[name][i], history[name][i + 1]
            pct = 100 * (after - before) / before
            (same_topic_pct if topic_of[name] == topic_of[w_name] else diff_topic_pct).append(pct)

    print("\none-shot interference on OTHER images (mean % change in recall):")
    print(f"  same topic:      {np.mean(same_topic_pct):+.2f}%  (n={len(same_topic_pct)})")
    print(f"  different topic: {np.mean(diff_topic_pct):+.2f}%  (n={len(diff_topic_pct)})")


if __name__ == "__main__":
    main()
