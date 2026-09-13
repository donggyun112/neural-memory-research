from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from neural_memory.contextbench_data import load_contextbench_episodes
from neural_memory.outcome_labels import hashed_text_features
from neural_memory.text_encoders import encode_with_gemma


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ContextBench trace/query features")
    parser.add_argument("--input", type=Path, default=Path("local-data/contextbench.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/contextbench.pt"))
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--positives-per-episode", type=int, default=4)
    parser.add_argument("--encoder", choices=("hash", "bge", "gemma"), default="hash")
    parser.add_argument("--model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling", choices=("last", "mean"), default="last")
    args = parser.parse_args()
    episodes = load_contextbench_episodes(
        args.input, positives_per_episode=args.positives_per_episode
    )
    if not episodes:
        raise RuntimeError("no ContextBench episodes found")
    texts = list(
        dict.fromkeys(
            text
            for episode in episodes
            for text in (
                episode.query_context,
                *(candidate.context for candidate in episode.candidates),
            )
        )
    )
    if args.encoder == "bge":
        from sentence_transformers import SentenceTransformer

        model_name = args.model or "BAAI/bge-small-en-v1.5"
        encoder = SentenceTransformer(model_name, device=args.device)
        encoded = encoder.encode(
            texts,
            batch_size=args.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).cpu()
        cache = dict(zip(texts, encoded, strict=True))
        feature_dim = encoded.shape[-1]
    elif args.encoder == "gemma":
        model_name = args.model or "google/gemma-3-270m"
        encoded = encode_with_gemma(
            texts,
            model_name=model_name,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
        )
        cache = dict(zip(texts, encoded, strict=True))
        feature_dim = encoded.shape[-1]
    else:
        model_name = "hash"
        cache = {
            text: hashed_text_features(text, dimension=args.feature_dim) for text in texts
        }
        feature_dim = args.feature_dim

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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "all_candidates": candidates,
            "all_queries": queries,
            "all_targets": targets,
            "all_masks": masks,
            "all_project_ids": projects,
            "encoder": args.encoder,
            "encoder_model": model_name,
            "pooling": args.pooling if args.encoder == "gemma" else "native",
        },
        args.output,
    )
    print(
        f"saved episodes={len(episodes)},projects={projects.unique().numel()},"
        f"candidates={int(masks.sum())},positive={int(targets[masks].sum())},"
        f"dim={feature_dim},encoder={args.encoder},raw_text=false to {args.output}"
    )


if __name__ == "__main__":
    main()
