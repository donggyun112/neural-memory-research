from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_memory.contextbench_data import load_contextbench_episodes
from neural_memory.text_encoders import encode_with_gemma_echo
from prepare_contextbench_gemma_layers import build_payload, unique_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ContextBench Gemma echo features")
    parser.add_argument("--input", type=Path, default=Path("local-data/contextbench.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/contextbench-gemma-echo.pt"))
    parser.add_argument("--model", default="google/gemma-3-270m")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--positives-per-episode", type=int, default=4)
    args = parser.parse_args()
    episodes = load_contextbench_episodes(
        args.input, positives_per_episode=args.positives_per_episode
    )
    if not episodes:
        raise RuntimeError("no ContextBench episodes found")
    texts = unique_texts(episodes)
    features = encode_with_gemma_echo(
        texts,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        layer=args.layer,
    )
    cache = dict(zip(texts, features, strict=True))
    payload = build_payload(
        episodes,
        cache,
        model_name=args.model,
        pooling="echo-mean",
        layer=args.layer,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved shape={tuple(features.shape)},layer={args.layer} to {args.output}")


if __name__ == "__main__":
    main()
