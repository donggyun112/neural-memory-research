from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from torch import Tensor

from neural_memory.contextbench_data import ContextEpisode, load_contextbench_episodes
from neural_memory.text_encoders import encode_gemma_layers


def unique_texts(episodes: list[ContextEpisode]) -> list[str]:
    return list(
        dict.fromkeys(
            text
            for episode in episodes
            for text in (
                episode.query_context,
                *(candidate.context for candidate in episode.candidates),
            )
        )
    )


def build_payload(
    episodes: list[ContextEpisode],
    cache: dict[str, Tensor],
    *,
    model_name: str,
    pooling: str,
    layer: int,
) -> dict[str, Tensor | str | int]:
    feature_dim = next(iter(cache.values())).shape[-1]
    max_candidates = max(len(episode.candidates) for episode in episodes)
    candidates = torch.zeros(len(episodes), max_candidates, feature_dim)
    queries = torch.zeros(len(episodes), feature_dim)
    targets = torch.zeros(len(episodes), max_candidates, dtype=torch.bool)
    masks = torch.zeros(len(episodes), max_candidates, dtype=torch.bool)
    projects = torch.zeros(len(episodes), dtype=torch.long)
    for row, episode in enumerate(episodes):
        queries[row] = cache[episode.query_context]
        for column, candidate in enumerate(episode.candidates):
            candidates[row, column] = cache[candidate.context]
            targets[row, column] = candidate.active
            masks[row, column] = True
        projects[row] = (
            int.from_bytes(hashlib.sha256(episode.project.encode()).digest()[:8], "big")
            % (2**63 - 1)
        )
    return {
        "all_candidates": candidates,
        "all_queries": queries,
        "all_targets": targets,
        "all_masks": masks,
        "all_project_ids": projects,
        "encoder": "gemma",
        "encoder_model": model_name,
        "pooling": pooling,
        "layer": layer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare several ContextBench Gemma layers")
    parser.add_argument("--input", type=Path, default=Path("local-data/contextbench.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/gemma-layers"))
    parser.add_argument("--model", default="google/gemma-3-270m")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling", choices=("last", "mean"), default="mean")
    parser.add_argument("--layers", default="3,6,9,12,15,18")
    parser.add_argument("--positives-per-episode", type=int, default=4)
    args = parser.parse_args()
    layers = tuple(int(value.strip()) for value in args.layers.split(",") if value.strip())
    episodes = load_contextbench_episodes(
        args.input, positives_per_episode=args.positives_per_episode
    )
    if not episodes:
        raise RuntimeError("no ContextBench episodes found")
    texts = unique_texts(episodes)
    encoded = encode_gemma_layers(
        texts,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        layers=layers,
        pooling=args.pooling,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for layer, features in encoded.items():
        cache = dict(zip(texts, features, strict=True))
        path = args.output_dir / f"contextbench-gemma-layer-{layer}-{args.pooling}.pt"
        torch.save(
            build_payload(
                episodes,
                cache,
                model_name=args.model,
                pooling=args.pooling,
                layer=layer,
            ),
            path,
        )
        print(f"saved layer={layer},shape={tuple(features.shape)} to {path}")


if __name__ == "__main__":
    main()
