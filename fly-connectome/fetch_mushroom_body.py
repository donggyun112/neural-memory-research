"""Pull the mushroom body circuit from neuPrint and save it as a sparse
adjacency matrix.

Phase 1 of an independent experiment line (see README.md): get the real
connectome loaded and verified. No simulation and no text input yet.

Defaults to MaleCNS v1.0 (Janelia/HHMI + Google Research + Cambridge + MRC
LMB, Cell 2026): 166,700 neurons, 125M+ synapses, the full CNS (brain + VNC)
of the male fly, released June 2026. This supersedes hemibrain (2020,
central-brain only, ~25k neurons) as the most complete public connectome.
Pass --dataset hemibrain:v1.2.1 to fall back to it.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import scipy.sparse as sp
from neuprint import Client

from connectome_io import SERVER, fetch_circuit as _fetch_circuit
from connectome_io import build_adjacency, fetch_circuit_neurons as _fetch_circuit_neurons, save_circuit

DATASET = "male-cns:v1.0"
MB_ROI = "MB(R)"

# Cell-class prefixes as named in the hemibrain connectome (Scheffer et al.
# 2020) and carried forward by Janelia FlyEM naming conventions. Not yet
# verified against MaleCNS v1.0's own type/ROI names -- check the first real
# fetch's summary output against known mushroom body counts before trusting it.
CLASS_PATTERNS = {
    "KC": re.compile(r"^KC"),
    "MBON": re.compile(r"^MBON"),
    "DAN": re.compile(r"^(PAM|PPL)"),
    "PN": re.compile(r"PN$"),
}


def cell_class(type_str: str | None) -> str:
    if not isinstance(type_str, str) or not type_str:
        return "other"
    for name, pattern in CLASS_PATTERNS.items():
        if pattern.search(type_str):
            return name
    return "other"


def fetch_circuit_neurons(client: Client, roi: str = MB_ROI):
    return _fetch_circuit_neurons(client, roi)


def fetch_circuit(client: Client, roi: str = MB_ROI):
    return _fetch_circuit(client, roi)


def summarize(neuron_df: pd.DataFrame, matrix: sp.csr_matrix) -> str:
    classes = neuron_df["type"].map(cell_class).value_counts()
    lines = [f"{roi_class}: {count}" for roi_class, count in classes.items()]
    lines.append(f"synapse edges: {matrix.nnz}, total weight: {int(matrix.sum())}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", default="MB(R)", help="ROI defining the circuit (default: right mushroom body)")
    parser.add_argument("--dataset", default=DATASET, help="neuPrint dataset (default: male-cns:v1.0)")
    parser.add_argument("--out-dir", default=Path(__file__).parent / "data", type=Path)
    parser.add_argument(
        "--token",
        default=os.environ.get("NEUPRINT_TOKEN"),
        help="neuPrint auth token. male-cns:v1.0 is a public read-only snapshot and may not need one -- try without it first.",
    )
    args = parser.parse_args()

    client = Client(SERVER, dataset=args.dataset, token=args.token)
    neuron_df, conn_df = fetch_circuit(client, roi=args.roi)
    matrix = build_adjacency(neuron_df, conn_df)

    save_circuit(args.out_dir, matrix, neuron_df)
    print(summarize(neuron_df, matrix))


if __name__ == "__main__":
    main()
