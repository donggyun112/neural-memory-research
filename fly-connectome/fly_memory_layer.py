"""The extracted, standalone long-term memory layer -- phase 25's best-known
mechanism (real KC->MBON synapses, dual fast/slow decay, each habituating
against its own pristine reference), with the connectome-fetching and
text/image front-end stripped away entirely.

Phase 19 of this experiment line (see README.md) asked whether the memory
mechanism could be lifted out and reused independent of the rest of this
pipeline; phase 26 updates the extracted layer to the best mechanism found
since (phase 25: habituated dual-store gave -41.5% load-under-interference
vs. phase 18's plain dual-store's -49.6%, at the same recognition strength
and topic discrimination -- see README). `FlyMemoryLayer` needs only a
KC x MBON weight matrix (plain numpy) to construct -- nothing about
neuPrint, embeddings, or sentence/image encoding. `from_mushroom_body_data_dir()`
is a convenience loader for this repo's cached fetch; anything else that can
produce a real KC x MBON weight matrix (or even a synthetic one) can use the
layer directly.

Interface deliberately mirrors experiments/neural-memory/neural_memory/*.py
(MemoryState/FastWeightState, write(...) -> new State rather than in-place
mutation, a batch-first leading dimension) so it can sit next to those
models without translation, while staying plain numpy -- no torch
dependency, per this line's explicit choice to keep the extraction
lightweight and independent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlyMemoryState:
    """Per-episode KC->MBON weights -- this array IS the memory, mirroring
    MemoryState/FastWeightState in neural_memory/*.py. Shape
    [batch, n_kc, n_mbon] for both `fast` and `slow`."""

    fast: np.ndarray
    slow: np.ndarray

    def copy(self) -> "FlyMemoryState":
        """Named to match MemoryState.detached() (no autograd graph in
        numpy, so there's nothing to detach from -- just an explicit copy)."""
        return FlyMemoryState(fast=self.fast.copy(), slow=self.slow.copy())


class FlyMemoryLayer:
    """Fixed real structure -- nothing here is gradient-trained. `write`
    returns a new `FlyMemoryState` rather than mutating in place, matching
    neural_memory/*.py's `write(...) -> State` convention (state.detached()
    there is state.copy() here)."""

    def __init__(self, kc_mbon_weights: np.ndarray, fast_eta: float = 0.5, slow_eta: float = 0.03):
        self.weights = np.asarray(kc_mbon_weights, dtype=float)  # [n_kc, n_mbon], the real connectome
        self.fast_eta = fast_eta
        self.slow_eta = slow_eta

    @property
    def n_kc(self) -> int:
        return self.weights.shape[0]

    @property
    def n_mbon(self) -> int:
        return self.weights.shape[1]

    def initial_state(self, batch_size: int = 1) -> FlyMemoryState:
        """One independent copy of the real connectome weights per batch
        item -- mirrors FastWeightMemory.initial_state(batch_size)."""
        fast = np.broadcast_to(self.weights, (batch_size, *self.weights.shape)).copy()
        return FlyMemoryState(fast=fast, slow=fast.copy())

    def write(self, state: FlyMemoryState, kc_code: np.ndarray) -> FlyMemoryState:
        """kc_code: [batch, n_kc]. Depresses each batch item's own
        KC->MBON rows for its currently active KCs (mirrors Owald et al.
        2015: reinforcement depresses, not potentiates, mushroom body
        output synapses), at two decay rates, each scaled by how fresh
        (undepressed relative to this layer's pristine `self.weights`)
        that row still is in its own store -- a never-touched KC gets the
        full eta, a KC already heavily depressed gets a progressively
        smaller one (habituation; phase 24/25)."""
        kc_code = self._as_batch(kc_code)
        self._check_batch(state, kc_code)
        active = kc_code > 0  # [batch, n_kc]
        initial_sum = self.weights.sum(axis=1)  # [n_kc]
        fast_freshness = state.fast.sum(axis=2) / initial_sum  # [batch, n_kc]
        slow_freshness = state.slow.sum(axis=2) / initial_sum
        fast_multiplier = np.where(active, 1.0 - self.fast_eta * fast_freshness, 1.0)
        slow_multiplier = np.where(active, 1.0 - self.slow_eta * slow_freshness, 1.0)
        fast = state.fast * fast_multiplier[:, :, None]
        slow = state.slow * slow_multiplier[:, :, None]
        return FlyMemoryState(fast=fast, slow=slow)

    def read(self, state: FlyMemoryState, kc_code: np.ndarray) -> np.ndarray:
        """kc_code: [batch, n_kc] -> [batch, n_mbon] recall vector (fast
        store + slow store, summed)."""
        kc_code = self._as_batch(kc_code)
        self._check_batch(state, kc_code)
        return np.einsum("bk,bkm->bm", kc_code, state.fast) + np.einsum("bk,bkm->bm", kc_code, state.slow)

    def _as_batch(self, kc_code: np.ndarray) -> np.ndarray:
        kc_code = np.asarray(kc_code, dtype=float)
        return kc_code[None, :] if kc_code.ndim == 1 else kc_code

    def _check_batch(self, state: FlyMemoryState, kc_code: np.ndarray) -> None:
        if state.fast.shape[0] != kc_code.shape[0]:
            raise ValueError(f"state batch size {state.fast.shape[0]} != kc_code batch size {kc_code.shape[0]}")
        if kc_code.shape[1] != self.n_kc:
            raise ValueError(f"kc_code has {kc_code.shape[1]} KC dims, layer expects {self.n_kc}")


def from_mushroom_body_data_dir(data_dir, fast_eta: float = 0.5, slow_eta: float = 0.03) -> FlyMemoryLayer:
    """Convenience loader for this repo's cached MB(R) fetch. Needs no
    neuPrint token or network access -- fetch_mushroom_body.py already did
    that; this only reads the local .parquet/.npz files it produced."""
    from pathlib import Path

    import pandas as pd
    import scipy.sparse as sp

    from fetch_mushroom_body import cell_class

    data_dir = Path(data_dir)
    neurons_df = pd.read_parquet(data_dir / "neurons.parquet")
    matrix = sp.load_npz(data_dir / "adjacency.npz")
    classes = neurons_df["type"].map(cell_class)
    kc_positions = np.flatnonzero((classes == "KC").to_numpy())
    mbon_positions = np.flatnonzero((classes == "MBON").to_numpy())
    weights = np.asarray(matrix[kc_positions][:, mbon_positions].todense(), dtype=float)
    return FlyMemoryLayer(weights, fast_eta=fast_eta, slow_eta=slow_eta)
