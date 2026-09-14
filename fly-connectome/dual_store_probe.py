"""Two KC->MBON stores with different decay time constants, read as a sum --
RESEARCH.md measurement 4 -- to see whether it resolves phase 17's trade-off:
a single flat write mode couldn't give both bounded capacity AND a legible
recognition signal at once.

Phase 18 (see README.md). fast store: large eta, changes a lot per write
(carries the recognition signal). slow store: small eta, barely moves per
write (carries capacity resistance). Recall = fast_drive + slow_drive.

Phase 22 adds --sweep: phase 18 picked fast_eta=0.5/slow_eta=0.03 as a first
guess: this grid-searches a few combinations and reports the same three
numbers for each, instead of asserting one pair is best.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from capacity_probe import TOPICS
from fetch_mushroom_body import cell_class
from generalization_probe import kc_code_for
from map_text_to_pn import load_pn_positions
from memory import kc_mbon_weights, mbon_drive, write


def dual_drive(fast_weights: np.ndarray, slow_weights: np.ndarray, kc_code: np.ndarray) -> float:
    return float(mbon_drive(fast_weights, kc_code).sum() + mbon_drive(slow_weights, kc_code).sum())


def dual_write(fast_weights: np.ndarray, slow_weights: np.ndarray, kc_code: np.ndarray, fast_eta: float, slow_eta: float) -> None:
    write(fast_weights, kc_code, eta=fast_eta)
    write(slow_weights, kc_code, eta=slow_eta)


def run_battery(
    matrix: sp.csr_matrix,
    kc_positions: np.ndarray,
    mbon_positions: np.ndarray,
    order: list[tuple[str, str]],
    codes: dict[str, np.ndarray],
    topic_of: dict[str, str],
    fast_eta: float,
    slow_eta: float,
) -> dict[str, float]:
    fast = kc_mbon_weights(matrix, kc_positions, mbon_positions)
    slow = fast.copy()

    first_sentence = order[0][1]
    exact_repeat_before = dual_drive(fast, slow, codes[first_sentence])
    dual_write(fast, slow, codes[first_sentence], fast_eta, slow_eta)
    exact_repeat_after = dual_drive(fast, slow, codes[first_sentence])
    exact_repeat_pct = 100 * (exact_repeat_after - exact_repeat_before) / exact_repeat_before

    fast = kc_mbon_weights(matrix, kc_positions, mbon_positions)
    slow = fast.copy()
    history = {s: [dual_drive(fast, slow, codes[s])] for _, s in order}
    for _, w_sentence in order:
        dual_write(fast, slow, codes[w_sentence], fast_eta, slow_eta)
        for _, s in order:
            history[s].append(dual_drive(fast, slow, codes[s]))

    same_topic_pct, diff_topic_pct = [], []
    for i, (_, w_sentence) in enumerate(order):
        for _, s in order:
            if s == w_sentence:
                continue
            before, after = history[s][i], history[s][i + 1]
            pct = 100 * (after - before) / before
            (same_topic_pct if topic_of[s] == topic_of[w_sentence] else diff_topic_pct).append(pct)

    trace = history[first_sentence]
    load_11_pct = 100 * (trace[12] - trace[0]) / trace[0]

    return {
        "exact_repeat_pct": exact_repeat_pct,
        "same_topic_pct": float(np.mean(same_topic_pct)),
        "diff_topic_pct": float(np.mean(diff_topic_pct)),
        "load_11_pct": load_11_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-eta", type=float, default=0.5)
    parser.add_argument("--slow-eta", type=float, default=0.03)
    parser.add_argument("--sweep", action="store_true", help="grid-search several fast/slow eta pairs")
    args = parser.parse_args()

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

    if not args.sweep:
        result = run_battery(matrix, kc_positions, mbon_positions, order, codes, topic_of, args.fast_eta, args.slow_eta)
        print(f"fast_eta={args.fast_eta}, slow_eta={args.slow_eta}\n")
        print(f"exact repeat:      {result['exact_repeat_pct']:+.1f}%")
        print(f"same topic:        {result['same_topic_pct']:+.2f}%")
        print(f"different topic:   {result['diff_topic_pct']:+.2f}%")
        print(f"load @ 11 writes:  {result['load_11_pct']:+.1f}%")
        return

    fast_etas = [0.3, 0.5, 0.7]
    slow_etas = [0.01, 0.03, 0.05, 0.1]
    print(f"{'fast_eta':>9} {'slow_eta':>9} {'exact_repeat':>13} {'load@11':>9} {'same/diff ratio':>17}")
    for fast_eta in fast_etas:
        for slow_eta in slow_etas:
            result = run_battery(matrix, kc_positions, mbon_positions, order, codes, topic_of, fast_eta, slow_eta)
            ratio = abs(result["same_topic_pct"]) / abs(result["diff_topic_pct"]) if result["diff_topic_pct"] else float("nan")
            print(f"{fast_eta:>9.2f} {slow_eta:>9.2f} {result['exact_repeat_pct']:>12.1f}% "
                  f"{result['load_11_pct']:>8.1f}% {ratio:>17.2f}")


if __name__ == "__main__":
    main()
