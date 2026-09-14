"""Write/recall plasticity on the real KC->MBON synapses -- the actual memory
mechanism, not just a per-sentence score.

Phase 5 of this experiment line (see README.md). Write = depress the
KC->MBON weights at the currently active KC set, mirroring Owald et al.
2015's finding that reinforcement depresses (not potentiates) mushroom body
output synapses; no per-compartment valence sign is invented since MBON
type names alone don't give us one. Recall = weighted MBON drive under the
current (possibly-depressed) weights.

Falsifiable claim: recalling the exact same KC code a second time, after its
own first write, must show a lower MBON drive than the first time -- the
weights it draws on were strictly reduced by that first write. That's the
minimal memory signature: state persists and changes what comes back.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fetch_mushroom_body import cell_class

DEFAULT_ETA = 0.3  # depression fraction per write


def class_positions(neurons_df: pd.DataFrame, cls: str) -> np.ndarray:
    return np.flatnonzero((neurons_df["type"].map(cell_class) == cls).to_numpy())


def kc_mbon_weights(matrix: sp.csr_matrix, kc_positions: np.ndarray, mbon_positions: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[kc_positions][:, mbon_positions].todense(), dtype=float)


def mbon_drive(weights: np.ndarray, kc_code: np.ndarray) -> np.ndarray:
    return kc_code @ weights


def write(weights: np.ndarray, kc_code: np.ndarray, eta: float = DEFAULT_ETA) -> None:
    """In-place depression of KC->MBON rows for the currently active KCs."""
    active = kc_code > 0
    weights[active, :] *= 1.0 - eta


def write_budgeted(weights: np.ndarray, kc_code: np.ndarray, eta: float = DEFAULT_ETA) -> None:
    """Same depression as write(), but rescales the whole matrix afterward to
    conserve total synaptic weight -- a fixed plasticity budget (Jeong et al.
    2021: potentiated inputs are preferentially recruited while the total
    recall-active population stays about constant), instead of every write
    eroding the total for free (phase 7's finding)."""
    total_before = weights.sum()
    write(weights, kc_code, eta=eta)
    total_after = weights.sum()
    if total_after > 0:
        weights *= total_before / total_after


def write_competitive(weights: np.ndarray, kc_code: np.ndarray, eta: float = DEFAULT_ETA) -> None:
    """A fixed depression budget shared across the currently active KCs,
    each getting a share proportional to its own activation (`a = sparse
    competition(...)`, `sum(a) = 1`, RESEARCH.md's licensed constraint --
    unlike write()/write_budgeted(), where every active row independently
    takes the same flat eta hit regardless of how many KCs are active, so a
    write's total damage grows with the number of active KCs instead of
    staying capped. Phase 7b's global rescale only partially fixed the
    unbounded-erosion-under-load problem this targets directly."""
    active = kc_code > 0
    if not active.any():
        return
    shares = kc_code[active] / kc_code[active].sum()  # sum(shares) = 1
    weights[active, :] *= (1.0 - eta * shares)[:, None]


def write_magnitude_weighted(weights: np.ndarray, kc_code: np.ndarray, eta: float = DEFAULT_ETA) -> None:
    """Scale each active row's depression by its own value relative to the
    strongest active row, instead of write()'s binary active-mask-only
    rule. Phase 31 found write() is otherwise blind to magnitude: when a
    small, convergent circuit routes every input to the same downstream
    set (only *which* rows are active differs for a large sparse
    population like KC, never for a small one), every code sharing that
    set gets depressed identically regardless of how different its real
    content is. This is what lets recall (a weighted dot product) and
    write (now also magnitude-sensitive) agree on what the code actually
    contains."""
    active = kc_code > 0
    if not active.any():
        return
    peak = kc_code[active].max()
    relative = kc_code[active] / peak if peak > 0 else np.zeros_like(kc_code[active])
    weights[active, :] *= (1.0 - eta * relative)[:, None]


def write_surprise_gated(weights: np.ndarray, initial_weights: np.ndarray, kc_code: np.ndarray, eta: float = DEFAULT_ETA) -> None:
    """Scale the depression by how "fresh" (undepressed relative to the
    pristine connectome) the currently active KCs still are -- a novel KC
    row gets the full write, a KC row already heavily depressed by past
    writes gets a progressively smaller one. This is habituation: repeated
    exposure to the same thing yields diminishing further encoding, instead
    of every write costing the same regardless of whether anything is left
    to change. Also self-limits erosion under unrelated load, the same way
    pushing fast_eta near 1 did in phase 22 -- an already-near-zero row has
    little left to lose."""
    active = kc_code > 0
    if not active.any():
        return
    freshness = weights[active, :].sum(axis=1) / initial_weights[active, :].sum(axis=1)
    effective_eta = eta * freshness
    weights[active, :] *= (1.0 - effective_eta)[:, None]


WRITE_MODES = {
    "unbudgeted": write,
    "budgeted": write_budgeted,
    "competitive": write_competitive,
    "magnitude_weighted": write_magnitude_weighted,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=Path(__file__).parent / "data", type=Path)
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--write-mode", choices=WRITE_MODES, default="unbudgeted")
    args = parser.parse_args()
    write_fn = WRITE_MODES[args.write_mode]

    neurons_df = pd.read_parquet(args.data_dir / "neurons.parquet")
    matrix = sp.load_npz(args.data_dir / "adjacency.npz")
    kc_data = np.load(args.data_dir / "kc_codes.npz")

    kc_positions = class_positions(neurons_df, "KC")
    mbon_positions = class_positions(neurons_df, "MBON")
    weights = kc_mbon_weights(matrix, kc_positions, mbon_positions)

    print(f"{len(kc_positions)} KC -> {len(mbon_positions)} MBON, eta={args.eta}, write_mode={args.write_mode}")
    seen: dict[str, float] = {}
    for sentence, code in zip(kc_data["sentences"], kc_data["kc_codes"]):
        sentence = str(sentence)
        drive = float(mbon_drive(weights, code).sum())
        write_fn(weights, code, eta=args.eta)

        preview = sentence if len(sentence) <= 60 else sentence[:57] + "..."
        if sentence in seen:
            delta = drive - seen[sentence]
            print(f"  drive={drive:8.2f}  (repeat, was {seen[sentence]:.2f}, Δ={delta:+.2f})  {preview}")
        else:
            print(f"  drive={drive:8.2f}  (first time)                    {preview}")
        seen[sentence] = drive


if __name__ == "__main__":
    main()
