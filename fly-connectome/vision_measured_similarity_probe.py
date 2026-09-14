"""Phase 16 found that human topic labels (coffee/dog/alarm) don't track
what this circuit's contrast-based codes actually consider similar (a
same-topic pair like cup-vs-beans overlapped LESS than the cross-topic
cup-vs-smoke_detector pair). This retests the real claim -- does
interference track *measured* KC overlap -- without that label mismatch:
rank all pairs by their actual overlap, and compare the single most-similar
pair against the single least-similar pair directly.

Phase 27 (see README.md). Reuses the same 6 real, locally-cached images
from vision_memory_probe.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fetch_optic_lobe import photoreceptor_class
from image_to_photoreceptors import build_stimulus, fetch_image
from kc_response import propagate, sparse_code
from memory import mbon_drive, write
from vision_memory_probe import IMAGE_TOPICS, KEEP_RATIO, lamina_code


def main() -> None:
    data_dir = "data_optic"
    neurons_df = pd.read_parquet(f"{data_dir}/neurons.parquet")
    matrix = sp.load_npz(f"{data_dir}/adjacency.npz")
    retinotopy = dict(np.load(f"{data_dir}/retinotopy.npz"))

    classes = neurons_df["type"].map(photoreceptor_class)
    other_mask = (classes == "other").to_numpy()
    other_positions = np.flatnonzero(other_mask)

    names_and_urls = [(name, url) for images in IMAGE_TOPICS.values() for name, url in images.items()]
    codes = {}
    for name, url in names_and_urls:
        image = fetch_image(url)
        stimulus = build_stimulus(image, neurons_df, retinotopy)
        codes[name] = lamina_code(stimulus, matrix, other_mask, KEEP_RATIO)

    active_sets = {name: set(np.flatnonzero(code)) for name, code in codes.items()}
    names = list(codes)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            union = active_sets[a] | active_sets[b]
            jaccard = len(active_sets[a] & active_sets[b]) / len(union) if union else 0.0
            pairs.append((jaccard, a, b))
    pairs.sort(key=lambda p: -p[0])

    most_similar = pairs[0]
    least_similar = pairs[-1]
    print("ranked by measured overlap (ignoring human topic labels):")
    for jaccard, a, b in pairs:
        print(f"  {jaccard:.3f}  {a} vs {b}")
    print(f"\nmost similar pair:  {most_similar[1]} vs {most_similar[2]} (overlap={most_similar[0]:.3f})")
    print(f"least similar pair: {least_similar[1]} vs {least_similar[2]} (overlap={least_similar[0]:.3f})")

    def one_shot_interference(pair) -> float:
        _, a, b = pair
        weights = np.asarray(matrix[other_positions][:, other_positions].todense(), dtype=float)
        before = float(mbon_drive(weights, codes[b]).sum())
        write(weights, codes[a], eta=0.3)
        after = float(mbon_drive(weights, codes[b]).sum())
        return 100 * (after - before) / before

    similar_effect = one_shot_interference(most_similar)
    dissimilar_effect = one_shot_interference(least_similar)
    print(f"\nwriting one image, effect on its partner's recall:")
    print(f"  most-similar pair:  {similar_effect:+.2f}%")
    print(f"  least-similar pair: {dissimilar_effect:+.2f}%")


if __name__ == "__main__":
    main()
