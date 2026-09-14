"""Re-run the phase 7 capacity/interference battery on real LongMemEval
document embeddings instead of hand-picked sentences -- grounding the
same-topic/different-topic test in the same real corpus
experiments/neural-memory's gist/surface work uses.

Phase 21 (see README.md). Deliberately does NOT attempt to reproduce
neural-memory's gist_recovery/surface_recovery metrics (analyze_load_and_gist.py):
those measure whether stored CONTENT (a value vector) can be reconstructed,
via an associative delta-rule store. Our KC->MBON mechanism is
depression-only -- it has no stored value vector to reconstruct, only
interference/recognition (can it tell something has been seen before). That
is a different kind of memory, not a worse implementation of the same one,
so gist/surface recovery isn't a well-defined number for it. What IS
comparable: does load/topic-structure behave the way phase 7/8 already
found, when the documents are real LongMemEval embeddings instead of ours.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from fetch_mushroom_body import cell_class
from kc_response import propagate, sparse_code
from map_text_to_pn import load_pn_positions, project_to_pn
from memory import kc_mbon_weights, mbon_drive, write

DEFAULT_FEATURES = (
    Path(__file__).resolve().parents[1] / "neural-memory" / "artifacts" / "longmemeval_deferred.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--data-dir", default=Path(__file__).parent / "data", type=Path)
    parser.add_argument("--episodes", type=int, default=4, help="LongMemEval episodes to use (8 candidates each)")
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--keep-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    candidates = payload["candidates"][: args.episodes].numpy()  # [episodes, 8, 384]
    print(f"loaded {candidates.shape[0]} real LongMemEval episodes x {candidates.shape[1]} "
          f"candidates, {candidates.shape[2]}-dim (from {args.features.name})\n")

    neurons_df = pd.read_parquet(args.data_dir / "neurons.parquet")
    matrix = sp.load_npz(args.data_dir / "adjacency.npz")
    pn_positions, _ = load_pn_positions(args.data_dir / "neurons.parquet")
    n_total = len(neurons_df)
    kc_mask = (neurons_df["type"].map(cell_class) == "KC").to_numpy()
    mbon_positions = np.flatnonzero((neurons_df["type"].map(cell_class) == "MBON").to_numpy())
    kc_positions = np.flatnonzero(kc_mask)

    embeddings = candidates.reshape(-1, candidates.shape[-1])  # [episodes*8, 384]
    projection_path = args.data_dir / "pn_projection.npy"
    projection = np.load(projection_path)
    if projection.shape[0] != embeddings.shape[1]:
        rng = np.random.default_rng(args.seed)
        projection = rng.standard_normal((embeddings.shape[1], len(pn_positions))) / np.sqrt(embeddings.shape[1])
        print(f"note: pn_projection.npy was fit for a different embedding dim; used a fresh "
              f"{embeddings.shape[1]}->{len(pn_positions)} random projection for this run instead.\n")

    pn_activations = project_to_pn(embeddings, projection)
    codes = []
    for activation in pn_activations:
        stimulus = np.zeros(n_total)
        stimulus[pn_positions] = activation
        drive = np.maximum(0.0, propagate(stimulus, matrix))
        codes.append(sparse_code(drive[kc_mask], args.keep_ratio))
    codes = np.array(codes)  # [episodes*8, n_kc]

    episode_of = [i // candidates.shape[1] for i in range(len(codes))]
    weights = kc_mbon_weights(matrix, kc_positions, mbon_positions)

    history = [float(mbon_drive(weights, code).sum()) for code in codes]
    trace = {i: [history[i]] for i in range(len(codes))}
    for w_i in range(len(codes)):
        write(weights, codes[w_i], eta=args.eta)
        for i in range(len(codes)):
            trace[i].append(float(mbon_drive(weights, codes[i]).sum()))

    same_topic_pct, diff_topic_pct = [], []
    for w_i in range(len(codes)):
        for i in range(len(codes)):
            if i == w_i:
                continue
            before, after = trace[i][w_i], trace[i][w_i + 1]
            pct = 100 * (after - before) / before
            (same_topic_pct if episode_of[i] == episode_of[w_i] else diff_topic_pct).append(pct)

    print("one-shot interference on OTHER real-document candidates:")
    print(f"  same episode:      {np.mean(same_topic_pct):+.2f}%  (n={len(same_topic_pct)})")
    print(f"  different episode: {np.mean(diff_topic_pct):+.2f}%  (n={len(diff_topic_pct)})")


if __name__ == "__main__":
    main()
