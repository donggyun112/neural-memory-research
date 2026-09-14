"""Pull the Johnston's organ / antennal mechanosensory circuit (AMMC) from
neuPrint and save it as a sparse adjacency matrix.

Phase 9 of this experiment line (see README.md). Unlike text, which has no
real transduction pathway into the fly nervous system, near-field sound /
antennal vibration does: physical vibration -> antenna -> Johnston's organ
(JO) mechanoreceptor neurons -> AMMC. Real named JO subgroups (A-F) exist in
MaleCNS v1.0's `type` field. Per Kamikouchi et al. 2009, JO-A/JO-B are
sound/vibration-responsive and JO-C/JO-E are gravity/wind (static
deflection)-responsive -- that citation is at the group level only; this
project does not have (and does not claim) verified frequency tuning for
the finer subtypes (JO-B1_a vs JO-B1_b, etc.).
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from neuprint import Client

from connectome_io import SERVER, build_adjacency, fetch_circuit, save_circuit
from fetch_mushroom_body import DATASET

AMMC_ROI = "AMMC(R)"

JO_GROUP_PATTERN = re.compile(r"^JO-([A-Z])")


def jo_group(type_str: str | None) -> str:
    """First letter after 'JO-' (A/B/C/E/F); 'other' for non-JO neurons and
    unclear-labeled JO neurons alike."""
    if not isinstance(type_str, str):
        return "other"
    match = JO_GROUP_PATTERN.match(type_str)
    return match.group(1) if match else "other"


def summarize(neuron_df, matrix) -> str:
    groups = neuron_df["type"].map(jo_group).value_counts()
    lines = [f"JO-{g}: {count}" if g != "other" else f"other: {count}" for g, count in groups.items()]
    lines.append(f"synapse edges: {matrix.nnz}, total weight: {int(matrix.sum())}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", default=AMMC_ROI, help="ROI defining the circuit (default: right AMMC)")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--out-dir", default=Path(__file__).parent / "data_ammc", type=Path)
    parser.add_argument("--token", default=os.environ.get("NEUPRINT_TOKEN"))
    args = parser.parse_args()

    client = Client(SERVER, dataset=args.dataset, token=args.token)
    neuron_df, conn_df = fetch_circuit(client, roi=args.roi)
    matrix = build_adjacency(neuron_df, conn_df)

    save_circuit(args.out_dir, matrix, neuron_df)
    print(summarize(neuron_df, matrix))


if __name__ == "__main__":
    main()
