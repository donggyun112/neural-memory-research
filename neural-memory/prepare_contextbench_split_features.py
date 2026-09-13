from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_memory.contextbench_data import load_contextbench_episodes
from neural_memory.text_encoders import encode_with_gemma


def split_context(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines[0], "\n".join(lines[1:])
    words = text.split()
    if len(words) < 2:
        return text, text
    boundary = max(1, len(words) // 2)
    return " ".join(words[:boundary]), " ".join(words[boundary:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporally split ContextBench features")
    parser.add_argument("--input", type=Path, default=Path("local-data/contextbench.jsonl"))
    parser.add_argument(
        "--base-features", type=Path, default=Path("artifacts/contextbench-gemma-mean.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/contextbench-gemma-split.pt")
    )
    parser.add_argument("--model", default="google/gemma-3-270m")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    episodes = load_contextbench_episodes(args.input)
    base = torch.load(args.base_features, map_location="cpu", weights_only=True)
    if len(episodes) != len(base["all_targets"]):
        raise ValueError("base artifact does not match ContextBench episodes")
    splits = [split_context(episode.query_context) for episode in episodes]
    texts = list(dict.fromkeys(text for pair in splits for text in pair))
    encoded = encode_with_gemma(
        texts,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling="mean",
    )
    cache = dict(zip(texts, encoded, strict=True))
    payload = dict(base)
    payload.update(
        {
            "all_write_queries": torch.stack([cache[write] for write, _ in splits]),
            "all_recall_queries": torch.stack([cache[recall] for _, recall in splits]),
            "split": "first-nonempty-line/rest",
            "split_max_length": args.max_length,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"saved episodes={len(episodes)},unique_split_texts={len(texts)},"
        f"dim={encoded.shape[-1]} to {args.output}"
    )


if __name__ == "__main__":
    main()
