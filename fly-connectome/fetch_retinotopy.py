"""Compute each photoreceptor's real position in the retinotopic map, from
the connectome's own synapse coordinates -- not soma location (often absent
for optic-lobe-only queries) but the centroid of each neuron's real synapses
within LA(R), which lamina cartridges arrange in the same 2D lattice as the
ommatidia they serve.

Phase 13 of this experiment line (see README.md). Fixes phase 12's honestly-
flagged gap: photoreceptor "order" was arbitrary, so image structure (edges,
shapes) couldn't survive the encoding. PCA reduces the 3D synapse centroids
to 2 axes -- the lamina is a roughly planar sheet, so this should recover
something close to the real 2D retinotopic layout without needing external
anatomical annotation.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from neuprint import Client, NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapses

from connectome_io import SERVER
from fetch_mushroom_body import DATASET
from fetch_optic_lobe import LAMINA_ROI


def fetch_photoreceptor_centroids(client: Client, roi: str) -> pd.DataFrame:
    """Per-bodyId mean (x, y, z) of each photoreceptor's synapses in `roi`."""
    criteria = NC(rois=[roi], type=r"^(R1-R6|R7|R8)", regex=True, status="Traced", client=client)
    synapses = fetch_synapses(criteria, SC(rois=roi, primary_only=True), client=client)
    return synapses.groupby("bodyId")[["x", "y", "z"]].mean().reset_index()


def pca_2d(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def normalize_unit(coords: np.ndarray) -> np.ndarray:
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (coords - lo) / span


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", default=LAMINA_ROI)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--out-dir", default=Path(__file__).parent / "data_optic", type=Path)
    parser.add_argument("--token", default=os.environ.get("NEUPRINT_TOKEN"))
    args = parser.parse_args()

    client = Client(SERVER, dataset=args.dataset, token=args.token)
    centroids = fetch_photoreceptor_centroids(client, args.roi)

    xyz = centroids[["x", "y", "z"]].to_numpy(dtype=float)
    uv = normalize_unit(pca_2d(xyz))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "retinotopy.npz",
        bodyId=centroids["bodyId"].to_numpy(),
        uv=uv,
    )

    # sanity check: a real ~planar sheet should have two large singular values
    # and one small one; report the ratio so a bad fit is visible, not silent.
    centered = xyz - xyz.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    print(f"{len(centroids)} photoreceptors with synapses in {args.roi}")
    print(f"singular values (x, y, z-ish axes): {singular_values}")
    print(f"planarity ratio (3rd / 2nd): {singular_values[2] / singular_values[1]:.3f} (near 0 = flat sheet)")


if __name__ == "__main__":
    main()
