"""Can the circuit tell a written item from one it has never seen?

Every mechanism comparison in this line so far reports two numbers: how far an
item's own MBON drive falls when it is written, and how far the first item has
fallen after eleven more writes. Neither answers the question a memory is for.
The load number is measured against the pristine weights, so it *contains* the
item's own write -- a mechanism that writes shallowly looks good on load for
that reason alone -- and neither number involves an item that was never written,
so nothing here distinguishes "the memory survived" from "everything sank less".

This probe writes a random half of the battery and then reads every sentence,
written and unwritten alike, scoring how separable the two groups are. It also
runs the control that has no plasticity at all: keep each written KC code and
score a probe by its best overlap with them. If storing the codes and comparing
them beats changing the synapses, then the plasticity is a lossy way to compute
an overlap that was already available.
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
from kc_response import propagate, sparse_code
from map_text_to_pn import load_pn_positions, project_to_pn
from memory import kc_mbon_weights, mbon_drive, write, write_surprise_gated

DEFAULT_FEATURES = (
    Path(__file__).resolve().parents[1] / "neural-memory" / "artifacts" / "longmemeval_deferred.pt"
)


def auc(written: np.ndarray, unwritten: np.ndarray) -> float:
    """Probability that a written item scores above an unwritten one.

    Ties count as half, so a mechanism that separates nothing reads 0.5 rather
    than an accidental 1.0 or 0.0.
    """
    if written.size == 0 or unwritten.size == 0:
        raise ValueError("both groups must be non-empty")
    comparisons = written[:, None] - unwritten[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def depression_scores(
    initial: np.ndarray, codes: list[np.ndarray], written: list[int], mode: str
) -> np.ndarray:
    """Run one write mode over the chosen items, then read every item.

    The score is how far each item's drive fell relative to its own pristine
    drive, so items that simply drive the MBONs harder do not dominate.
    """
    pristine = np.array([float(mbon_drive(initial, code).sum()) for code in codes])
    if mode in ("unbudgeted", "saturating"):
        weights = initial.copy()
        # The tuned pair sets the fast store's eta to 0.99, which nearly zeroes
        # every written row. If that alone explains its result, the second store
        # is doing no work and calling the mechanism "dual" is a misnomer.
        eta = 0.3 if mode == "unbudgeted" else 0.99
        for index in written:
            write(weights, codes[index], eta=eta)
        after = np.array([float(mbon_drive(weights, code).sum()) for code in codes])
    elif mode == "habituation":
        weights = initial.copy()
        for index in written:
            write_surprise_gated(weights, initial, codes[index], eta=0.3)
        after = np.array([float(mbon_drive(weights, code).sum()) for code in codes])
    else:
        fast_eta, slow_eta, habituated = {
            "dual": (0.5, 0.03, False),
            "dual_tuned": (0.99, 0.001, False),
            "habituated_dual": (0.5, 0.03, True),
        }[mode]
        fast, slow = initial.copy(), initial.copy()
        for index in written:
            code = codes[index]
            active = code > 0
            if not active.any():
                continue
            if habituated:
                fast_share = fast[active, :].sum(axis=1) / initial[active, :].sum(axis=1)
                slow_share = slow[active, :].sum(axis=1) / initial[active, :].sum(axis=1)
            else:
                fast_share = slow_share = np.ones(int(active.sum()))
            fast[active, :] *= (1.0 - fast_eta * fast_share)[:, None]
            slow[active, :] *= (1.0 - slow_eta * slow_share)[:, None]
        after = np.array(
            [
                float(mbon_drive(fast, code).sum() + mbon_drive(slow, code).sum())
                for code in codes
            ]
        )
        pristine = pristine * 2.0
    # Written items should have fallen further, so the score is the size of the
    # fall and larger means "more likely to have been written".
    return -(after - pristine) / pristine


def overlap_scores(codes: list[np.ndarray], written: list[int]) -> np.ndarray:
    """No plasticity at all: keep the written codes and score by best overlap."""
    stored = np.stack([codes[index] for index in written])
    norms = np.linalg.norm(stored, axis=1)
    scores = []
    for code in codes:
        size = np.linalg.norm(code)
        if size == 0:
            scores.append(0.0)
            continue
        scores.append(float((stored @ code / (norms * size)).max()))
    return np.array(scores)


def longmemeval_codes(
    features: Path,
    documents: int,
    data_dir: Path,
    matrix: sp.csr_matrix,
    pn_positions: np.ndarray,
    n_total: int,
    kc_mask: np.ndarray,
    keep_ratio: float,
    seed: int,
) -> list[np.ndarray]:
    """Real corpus documents, enough of them to push the store past its capacity."""
    import torch

    payload = torch.load(features, map_location="cpu", weights_only=True)
    embeddings = payload["candidates"].flatten(0, 1).numpy()[:documents]
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal(
        (embeddings.shape[1], len(pn_positions))
    ) / np.sqrt(embeddings.shape[1])
    codes = []
    for activation in project_to_pn(embeddings, projection):
        stimulus = np.zeros(n_total)
        stimulus[pn_positions] = activation
        codes.append(sparse_code(np.maximum(0.0, propagate(stimulus, matrix))[kc_mask], keep_ratio))
    return codes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=Path(__file__).parent / "data", type=Path)
    parser.add_argument("--source", choices=("topics", "longmemeval"), default="longmemeval")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--loads", default="4,8,16,32,64,128")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--keep-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    neurons_df = pd.read_parquet(args.data_dir / "neurons.parquet")
    matrix = sp.load_npz(args.data_dir / "adjacency.npz")
    pn_positions, _ = load_pn_positions(args.data_dir / "neurons.parquet")
    n_total = len(neurons_df)
    kc_mask = (neurons_df["type"].map(cell_class) == "KC").to_numpy()
    kc_positions = np.flatnonzero(kc_mask)
    mbon_positions = np.flatnonzero((neurons_df["type"].map(cell_class) == "MBON").to_numpy())
    initial = kc_mbon_weights(matrix, kc_positions, mbon_positions)

    loads = [int(value) for value in args.loads.split(",")]
    if args.source == "topics":
        projection = np.load(args.data_dir / "pn_projection.npy")
        codes = [
            kc_code_for(s, projection, matrix, pn_positions, n_total, kc_mask, args.keep_ratio)
            for sents in TOPICS.values()
            for s in sents
        ]
        loads = [value for value in loads if 2 * value <= len(codes)]
    else:
        codes = longmemeval_codes(
            args.features,
            2 * max(loads),
            args.data_dir,
            matrix,
            pn_positions,
            n_total,
            kc_mask,
            args.keep_ratio,
            args.seed,
        )

    modes = [
        "unbudgeted",
        "saturating",
        "dual",
        "dual_tuned",
        "habituation",
        "habituated_dual",
    ]
    names = [*modes, "stored_codes"]
    print(
        f"source={args.source}, {len(codes)} documents available, {args.trials} trials per load"
    )
    print(
        "AUC that a written item scores above an unwritten one; 0.5 is no separation.\n"
        "stored_codes keeps one code per written item, so its memory grows with the load;\n"
        "every other row is a fixed-size synaptic state, which is what the load is testing."
    )
    header = ' '.join(f"{name:>17}" for name in names)
    print(f"{'written':>8} {header}")
    rng = np.random.default_rng(args.seed)
    for load in loads:
        if 2 * load > len(codes):
            continue
        rows: dict[str, list[float]] = {name: [] for name in names}
        for _ in range(args.trials):
            chosen = rng.choice(len(codes), size=2 * load, replace=False)
            written, held = sorted(chosen[:load].tolist()), sorted(chosen[load:].tolist())
            for mode in modes:
                scores = depression_scores(initial, codes, written, mode)
                rows[mode].append(auc(scores[written], scores[held]))
            stored = overlap_scores(codes, written)
            rows["stored_codes"].append(auc(stored[written], stored[held]))
        cells = ' '.join(
            f"{np.mean(rows[name]):11.4f}±{np.std(rows[name]):.3f}" for name in names
        )
        print(f"{load:>8} {cells}")


if __name__ == "__main__":
    main()
