"""A grounded text->PN converter: when a sentence names a specific, well-
studied odorant, activate the *real* glomerulus known (from published
electrophysiology) to respond to it, instead of the random projection
map_text_to_pn.py falls back to for everything else.

Phase 10 of this experiment line (see README.md). Small and deliberately
conservative -- only odor/glomerulus pairs with strong, specific literature
support are included. Confirmed against the loaded MB(R) circuit: every
glomerulus below has a matching real PN type (e.g. "DA2_lPN").
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from map_text_to_pn import (
    encode_sentences,
    get_or_create_projection,
    project_to_pn,
    split_sentences,
)

# Neutral filler text, used only to measure the random-fallback path's typical
# per-neuron activation scale -- so a grounded activation can be put on the
# same footing instead of an arbitrary fixed magnitude.
_REFERENCE_SENTENCES = [
    "The train arrived at the station on time.",
    "She opened the window to let in some air.",
    "The book was left on the kitchen table.",
    "A car passed slowly down the street.",
    "He finished his homework before dinner.",
    "The lights flickered during the storm.",
    "They walked along the river in the evening.",
    "The clock on the wall stopped at noon.",
]

# keyword -> (glomerulus prefix, citation). Only well-established, specific
# odorant-to-glomerulus pairings -- not a general odor vocabulary.
ODOR_GLOMERULI: dict[str, tuple[str, str]] = {
    "geosmin": ("DA2", "Stensmyr et al. 2012 (Cell): geosmin -> glomerulus DA2, innately aversive."),
    "vinegar": ("DM1", "Hallem & Carlson 2006: acetic acid (vinegar) strongly drives Or42b -> DM1."),
    "acetic acid": ("DM1", "Hallem & Carlson 2006: acetic acid strongly drives Or42b -> DM1."),
    "carbon dioxide": ("V", "Suh et al. 2004 (Nature): CO2 -> ab1C neurons -> glomerulus V."),
    "co2": ("V", "Suh et al. 2004 (Nature): CO2 -> ab1C neurons -> glomerulus V."),
    "pheromone": ("DA1", "Kurtovic et al. 2007: cVA pheromone -> Or67d neurons -> glomerulus DA1."),
    "cva": ("DA1", "Kurtovic et al. 2007: cVA pheromone -> Or67d neurons -> glomerulus DA1."),
    "ethanol": ("DM2", "Hallem & Carlson 2006: ethanol strongly drives Or59b -> glomerulus DM2."),
    "ethyl butyrate": ("DM2", "Hallem & Carlson 2006: ethyl butyrate drives Or22a -> glomerulus DM2."),
}

_WORD_BOUNDARY = {k: re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE) for k in ODOR_GLOMERULI}


def find_odor_match(sentence: str) -> tuple[str, str, str] | None:
    """First matching (keyword, glomerulus, citation), or None."""
    for keyword, (glomerulus, citation) in ODOR_GLOMERULI.items():
        if _WORD_BOUNDARY[keyword].search(sentence):
            return keyword, glomerulus, citation
    return None


def glomerulus_pn_mask(pn_ids_in_order: np.ndarray, neurons_df: pd.DataFrame, glomerulus: str) -> np.ndarray:
    """Boolean mask, in the same order as `pn_ids_in_order`, of PNs whose
    type belongs to this glomerulus (type starts with "<glomerulus>_")."""
    type_by_id = neurons_df.set_index("bodyId")["type"]
    types = type_by_id.loc[pn_ids_in_order].to_numpy()
    prefix = f"{glomerulus}_"
    return np.array([isinstance(t, str) and t.startswith(prefix) for t in types])


def grounded_activation(pn_ids_in_order: np.ndarray, neurons_df: pd.DataFrame, glomerulus: str, magnitude: float) -> np.ndarray:
    mask = glomerulus_pn_mask(pn_ids_in_order, neurons_df, glomerulus)
    activation = np.zeros(len(pn_ids_in_order))
    activation[mask] = magnitude
    return activation


def reference_pn_magnitude(projection: np.ndarray) -> float:
    """Mean nonzero PN activation the random-fallback path produces on
    ordinary text -- the scale a grounded activation should match, instead
    of an arbitrary fixed value."""
    embeddings = encode_sentences(_REFERENCE_SENTENCES)
    activations = project_to_pn(embeddings, projection)
    nonzero = activations[activations > 0]
    return float(nonzero.mean()) if len(nonzero) else 1.0


def embed_sentences(
    sentences: list[str], neurons_df: pd.DataFrame, pn_ids: np.ndarray, projection: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Returns (activations, methods) where methods[i] is either
    "grounded:<glomerulus>" or "random-fallback"."""
    magnitude = reference_pn_magnitude(projection)
    activations = np.zeros((len(sentences), len(pn_ids)))
    methods = []
    random_sentences, random_rows = [], []
    for i, sentence in enumerate(sentences):
        match = find_odor_match(sentence)
        if match:
            keyword, glomerulus, _ = match
            activations[i] = grounded_activation(pn_ids, neurons_df, glomerulus, magnitude)
            methods.append(f"grounded:{glomerulus} ({keyword!r})")
        else:
            random_sentences.append(sentence)
            random_rows.append(i)
            methods.append("random-fallback")
    if random_sentences:
        embeddings = encode_sentences(random_sentences)
        activations[random_rows] = project_to_pn(embeddings, projection)
    return activations, methods


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--data-dir", default=Path(__file__).parent / "data", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    text = args.text_file.read_text() if args.text_file else args.text
    if not text:
        raise SystemExit("Pass --text or --text-file")
    sentences = split_sentences(text)

    from map_text_to_pn import load_pn_positions

    neurons_df = pd.read_parquet(args.data_dir / "neurons.parquet")
    pn_positions, pn_ids = load_pn_positions(args.data_dir / "neurons.parquet")

    embeddings_dim = 384  # all-MiniLM-L6-v2
    projection = get_or_create_projection(args.data_dir / "pn_projection.npy", embeddings_dim, len(pn_ids), seed=args.seed)

    activations, methods = embed_sentences(sentences, neurons_df, pn_ids, projection)

    out_path = args.data_dir / "pn_activations.npz"
    np.savez(
        out_path,
        activations=activations,
        pn_positions=pn_positions,
        pn_ids=pn_ids,
        sentences=np.array(sentences),
        methods=np.array(methods),
    )

    for sentence, method, activation in zip(sentences, methods, activations):
        active = int((activation > 0).sum())
        preview = sentence if len(sentence) <= 55 else sentence[:52] + "..."
        print(f"  [{method:28s}] {active:3d}/{len(pn_ids)} active  {preview}")


if __name__ == "__main__":
    main()
