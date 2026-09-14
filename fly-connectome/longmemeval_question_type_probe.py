"""Phase 21's same/different-episode test found no discrimination, likely
because LongMemEval's 8 candidates-per-episode are deliberately diverse
distractor sessions, not a topic cluster. This regroups by the real
`question_type` category (single-session-user, multi-session,
knowledge-update, temporal-reasoning, etc.) that `longmemeval_deferred.pt`
already carries -- a genuine external label, not one this project invented.

Phase 35 (see README.md). Same real PN->KC pipeline as phase 21.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

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
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--keep-ratio", type=float, default=0.10)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    all_types = payload["question_type"].numpy()
    rng = np.random.default_rng(0)
    selected = rng.choice(len(all_types), size=min(args.episodes, len(all_types)), replace=False)
    candidates = payload["candidates"][selected].numpy()  # [episodes, 8, 384]
    question_type = all_types[selected]
    type_names = payload["question_type_names"]
    print(f"loaded {candidates.shape[0]} episodes, question types present: "
          f"{sorted(set(type_names[t] for t in question_type))}\n")

    neurons_df = pd.read_parquet(args.data_dir / "neurons.parquet")
    matrix = sp.load_npz(args.data_dir / "adjacency.npz")
    pn_positions, _ = load_pn_positions(args.data_dir / "neurons.parquet")
    n_total = len(neurons_df)
    kc_mask = (neurons_df["type"].map(cell_class) == "KC").to_numpy()
    mbon_positions = np.flatnonzero((neurons_df["type"].map(cell_class) == "MBON").to_numpy())
    kc_positions = np.flatnonzero(kc_mask)

    embeddings = candidates.reshape(-1, candidates.shape[-1])
    projection = np.load(args.data_dir / "pn_projection.npy")
    pn_activations = project_to_pn(embeddings, projection)

    codes = []
    for activation in pn_activations:
        stimulus = np.zeros(n_total)
        stimulus[pn_positions] = activation
        drive = np.maximum(0.0, propagate(stimulus, matrix))
        codes.append(sparse_code(drive[kc_mask], args.keep_ratio))
    codes = np.array(codes)

    # each candidate inherits its episode's real question_type label
    type_of = [type_names[question_type[i // candidates.shape[1]]] for i in range(len(codes))]

    weights = kc_mbon_weights(matrix, kc_positions, mbon_positions)
    history = [float(mbon_drive(weights, code).sum()) for code in codes]
    trace = {i: [history[i]] for i in range(len(codes))}
    for w_i in range(len(codes)):
        write(weights, codes[w_i], eta=args.eta)
        for i in range(len(codes)):
            trace[i].append(float(mbon_drive(weights, codes[i]).sum()))

    same_type_pct, diff_type_pct = [], []
    per_type_pairs = defaultdict(list)
    for w_i in range(len(codes)):
        for i in range(len(codes)):
            if i == w_i:
                continue
            before, after = trace[i][w_i], trace[i][w_i + 1]
            pct = 100 * (after - before) / before
            if type_of[i] == type_of[w_i]:
                same_type_pct.append(pct)
                per_type_pairs[type_of[i]].append(pct)
            else:
                diff_type_pct.append(pct)

    print("one-shot interference on OTHER real candidates, grouped by real question_type:")
    print(f"  same type:      {np.mean(same_type_pct):+.2f}%  (n={len(same_type_pct)})")
    print(f"  different type: {np.mean(diff_type_pct):+.2f}%  (n={len(diff_type_pct)})")
    print("\nper-type breakdown:")
    for type_name, values in per_type_pairs.items():
        print(f"  {type_name:24s}: {np.mean(values):+.2f}%  (n={len(values)})")


if __name__ == "__main__":
    main()
