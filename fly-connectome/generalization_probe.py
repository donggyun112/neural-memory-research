"""Does writing one sentence affect recall of a *different but similar*
sentence, or only an exact repeat?

Phase 6 probe (see README.md). memory.py only showed exact-repeat recognition
(same string -> same KC code -> guaranteed depression). This measures
whether a paraphrase (different words, same event) shares enough of the KC
code to be affected too, versus an unrelated sentence used as a control.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fetch_mushroom_body import cell_class
from kc_response import class_mask, propagate, scatter, sparse_code
from map_text_to_pn import encode_sentences, load_pn_positions, project_to_pn
from memory import kc_mbon_weights, mbon_drive, write


def kc_code_for(sentence: str, projection: np.ndarray, matrix: sp.csr_matrix,
                 pn_positions: np.ndarray, n_total: int, kc_mask: np.ndarray, keep_ratio: float) -> np.ndarray:
    embedding = encode_sentences([sentence])
    activation = project_to_pn(embedding, projection)[0]
    stimulus = scatter(activation, pn_positions, n_total)
    drive = np.maximum(0.0, propagate(stimulus, matrix))
    return sparse_code(drive[kc_mask], keep_ratio)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=Path(__file__).parent / "data", type=Path)
    parser.add_argument("--original", default="The fire alarm rang loudly in the empty hallway.")
    parser.add_argument("--paraphrase", default="A loud alarm blared through the deserted corridor.")
    parser.add_argument("--control", default="A cup of coffee sat cold on the desk.")
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--keep-ratio", type=float, default=0.10)
    args = parser.parse_args()

    neurons_df = pd.read_parquet(args.data_dir / "neurons.parquet")
    matrix = sp.load_npz(args.data_dir / "adjacency.npz")
    projection = np.load(args.data_dir / "pn_projection.npy")
    pn_positions, _ = load_pn_positions(args.data_dir / "neurons.parquet")
    n_total = len(neurons_df)
    kc_mask = class_mask(neurons_df, "KC")

    def code(sentence: str) -> np.ndarray:
        return kc_code_for(sentence, projection, matrix, pn_positions, n_total, kc_mask, args.keep_ratio)

    original_code = code(args.original)
    paraphrase_code = code(args.paraphrase)
    control_code = code(args.control)

    overlap_paraphrase = len(set(np.flatnonzero(original_code)) & set(np.flatnonzero(paraphrase_code)))
    overlap_control = len(set(np.flatnonzero(original_code)) & set(np.flatnonzero(control_code)))
    n_active = int((original_code > 0).sum())
    print(f"KC overlap with original ({n_active} active): paraphrase={overlap_paraphrase}, control={overlap_control}")

    kc_positions = np.flatnonzero(kc_mask)
    mbon_positions = np.flatnonzero((neurons_df["type"].map(cell_class) == "MBON").to_numpy())
    weights = kc_mbon_weights(matrix, kc_positions, mbon_positions)

    baseline_paraphrase = float(mbon_drive(weights, paraphrase_code).sum())
    baseline_control = float(mbon_drive(weights, control_code).sum())

    write(weights, original_code, eta=args.eta)

    after_paraphrase = float(mbon_drive(weights, paraphrase_code).sum())
    after_control = float(mbon_drive(weights, control_code).sum())

    print(f"paraphrase drive: {baseline_paraphrase:.2f} -> {after_paraphrase:.2f} "
          f"(Δ={after_paraphrase - baseline_paraphrase:+.2f}, {100 * (after_paraphrase - baseline_paraphrase) / baseline_paraphrase:+.1f}%)")
    print(f"control drive:    {baseline_control:.2f} -> {after_control:.2f} "
          f"(Δ={after_control - baseline_control:+.2f}, {100 * (after_control - baseline_control) / baseline_control:+.1f}%)")


if __name__ == "__main__":
    main()
