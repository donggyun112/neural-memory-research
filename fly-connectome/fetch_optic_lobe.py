"""Pull the lamina (LA) circuit -- where photoreceptor axons actually
terminate -- from neuPrint and save it as a sparse adjacency matrix.

Phase 11 of this experiment line (see README.md). A second real
transduction pathway alongside Johnston's organ (phase 9): light intensity
-> photoreceptor membrane potential is real biophysics, unlike the random
projection text needed. DOOMFLY (github.com/nftechie/doomfly) already
proved out frame -> R1-R6/R8 activation on this exact connectome. LA(R) is
where R1-R8 axons terminate, a manageable ~7k-neuron circuit (ME(R), the
next stage, is ~43k -- too big for a first step).
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from neuprint import Client

from connectome_io import SERVER, build_adjacency, fetch_circuit, save_circuit
from fetch_mushroom_body import DATASET

LAMINA_ROI = "LA(R)"

PHOTORECEPTOR_PATTERN = re.compile(r"^(R1-R6|R7R8|R[78][a-z]?)")


def photoreceptor_class(type_str: str | None) -> str:
    """R1-R6 (brightness), R7/R8 subtypes (color/UV -- y=yellow, p=pale,
    d=dorsal rim), or 'other' for downstream lamina neurons (L1-L5 etc.)."""
    if not isinstance(type_str, str):
        return "other"
    match = PHOTORECEPTOR_PATTERN.match(type_str)
    return match.group(0) if match else "other"


def summarize(neuron_df, matrix) -> str:
    classes = neuron_df["type"].map(photoreceptor_class).value_counts()
    lines = [f"{name}: {count}" for name, count in classes.items()]
    lines.append(f"synapse edges: {matrix.nnz}, total weight: {int(matrix.sum())}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", default=LAMINA_ROI, help="ROI defining the circuit (default: right lamina)")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--out-dir", default=Path(__file__).parent / "data_optic", type=Path)
    parser.add_argument("--token", default=os.environ.get("NEUPRINT_TOKEN"))
    args = parser.parse_args()

    client = Client(SERVER, dataset=args.dataset, token=args.token)
    neuron_df, conn_df = fetch_circuit(client, roi=args.roi)
    matrix = build_adjacency(neuron_df, conn_df)

    save_circuit(args.out_dir, matrix, neuron_df)
    print(summarize(neuron_df, matrix))


if __name__ == "__main__":
    main()
