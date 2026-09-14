"""Map text, sentence by sentence, onto the real PN neurons loaded by
fetch_mushroom_body.py.

Phase 2 of this experiment line (see README.md): get a stimulus signal onto
the real input population. No KC/downstream dynamics and no "stimulus point"
definition yet -- just a PN activation vector per sentence.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_mushroom_body import cell_class

DEFAULT_MODEL = "all-MiniLM-L6-v2"

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?\n])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


def load_pn_positions(neurons_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Row/column positions of PN neurons in the adjacency matrix, in matrix
    order (NOT sorted by bodyId -- position is what lets a later step index
    straight into the matrix built by fetch_mushroom_body.py)."""
    neurons_df = pd.read_parquet(neurons_path)
    mask = (neurons_df["type"].map(cell_class) == "PN").to_numpy()
    positions = np.flatnonzero(mask)
    pn_ids = neurons_df["bodyId"].to_numpy()[positions]
    return positions, pn_ids


def random_projection(embedding_dim: int, n_targets: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((embedding_dim, n_targets)) / np.sqrt(embedding_dim)


def get_or_create_projection(path: Path, embedding_dim: int, n_targets: int, seed: int = 0) -> np.ndarray:
    """Fixed across runs once created -- the same sentence must land on the
    same simulated PN pattern every time, or nothing downstream is comparable."""
    if path.exists():
        matrix = np.load(path)
        if matrix.shape == (embedding_dim, n_targets):
            return matrix
    matrix = random_projection(embedding_dim, n_targets, seed=seed)
    np.save(path, matrix)
    return matrix


def project_to_pn(embeddings: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Firing rates can't be negative, so clip at zero after the projection."""
    return np.maximum(0.0, embeddings @ projection)


def encode_sentences(sentences: list[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(sentences))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Text to map, as a literal string")
    parser.add_argument("--text-file", type=Path, help="Text file to map (overrides --text)")
    parser.add_argument("--data-dir", default=Path(__file__).parent / "data", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.text_file:
        text = args.text_file.read_text()
    elif args.text:
        text = args.text
    else:
        raise SystemExit("Pass --text or --text-file")

    sentences = split_sentences(text)
    if not sentences:
        raise SystemExit("No sentences found in the input")

    pn_positions, pn_ids = load_pn_positions(args.data_dir / "neurons.parquet")
    embeddings = encode_sentences(sentences, model_name=args.model)
    projection = get_or_create_projection(
        args.data_dir / "pn_projection.npy", embeddings.shape[1], len(pn_ids), seed=args.seed
    )
    activations = project_to_pn(embeddings, projection)

    out_path = args.data_dir / "pn_activations.npz"
    np.savez(
        out_path,
        activations=activations,
        pn_positions=pn_positions,
        pn_ids=pn_ids,
        sentences=np.array(sentences),
    )

    print(f"{len(sentences)} sentences -> {len(pn_ids)} PN neurons")
    for sentence, activation in zip(sentences, activations):
        active = int((activation > 0).sum())
        preview = sentence if len(sentence) <= 60 else sentence[:57] + "..."
        print(f"  [{active:3d}/{len(pn_ids)} active, mean={activation.mean():.3f}] {preview}")


if __name__ == "__main__":
    main()
