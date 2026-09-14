"""Shared neuPrint fetch/save helpers, used by fetch_mushroom_body.py and
fetch_johnstons_organ.py to pull a named circuit's real connectivity."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import scipy.sparse as sp
from neuprint import Client, NeuronCriteria as NC, fetch_adjacencies, fetch_neurons

SERVER = "neuprint.janelia.org"
DEFAULT_DATASET = "male-cns:v1.0"


def fetch_circuit_neurons(client: Client, roi: str) -> pd.DataFrame:
    """Every traced neuron touching `roi` (cheap: no connection query)."""
    neuron_df, _ = fetch_neurons(NC(rois=[roi], status="Traced", client=client))
    return neuron_df


def fetch_circuit(client: Client, roi: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(neuron_df, conn_df) for every traced neuron touching `roi`, restricted
    to connections within that same neuron set."""
    neuron_df = fetch_circuit_neurons(client, roi=roi)
    body_ids = neuron_df["bodyId"].tolist()
    _, conn_df = fetch_adjacencies(body_ids, body_ids, omit_rois=True, client=client)
    return neuron_df, conn_df


def build_adjacency(neuron_df: pd.DataFrame, conn_df: pd.DataFrame) -> sp.csr_matrix:
    """Sparse (n_neurons x n_neurons) weighted adjacency, indexed by position
    in `neuron_df` (row order, NOT sorted by bodyId)."""
    index = {body_id: i for i, body_id in enumerate(neuron_df["bodyId"])}
    rows = conn_df["bodyId_pre"].map(index).to_numpy()
    cols = conn_df["bodyId_post"].map(index).to_numpy()
    weights = conn_df["weight"].to_numpy()
    n = len(neuron_df)
    return sp.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()


def save_circuit(out_dir: Path, matrix: sp.csr_matrix, neuron_df: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "adjacency.npz", matrix)
    neuron_df.to_parquet(out_dir / "neurons.parquet")
