from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_memory.longmemeval import iter_longmemeval_updates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare knowledge-update features where both statements stay in the pool"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/longmemeval_s_cleaned.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/longmemeval_updates.pt")
    )
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    episodes = list(iter_longmemeval_updates(args.input, candidates=args.candidates))
    if not episodes:
        raise RuntimeError("no knowledge-update episodes were produced")
    if any(not episode.answer.strip() for episode in episodes):
        raise RuntimeError("every update episode needs a gold answer")
    texts = [
        text
        for episode in episodes
        for text in (*episode.candidates, episode.question, episode.answer)
    ]
    encoder = SentenceTransformer(args.model, device=args.device)
    encoded = encoder.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).cpu()

    stride = args.candidates + 2
    rows = range(len(episodes))
    output = {
        "source": "LongMemEval-S cleaned, knowledge-update",
        "model": args.model,
        "candidates": torch.stack(
            [encoded[index * stride : index * stride + args.candidates] for index in rows]
        ),
        "query": torch.stack([encoded[index * stride + args.candidates] for index in rows]),
        "answer": torch.stack([encoded[index * stride + args.candidates + 1] for index in rows]),
        # The target is the statement that supersedes; the other one is what a
        # topic-matching reader is expected to confuse it with.
        "targets": torch.tensor(
            [episode.target_offset for episode in episodes], dtype=torch.long
        ),
        "superseded": torch.tensor(
            [episode.superseded_offset for episode in episodes], dtype=torch.long
        ),
        "lengths": torch.tensor(
            [[float(len(text)) for text in episode.candidates] for episode in episodes]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"saved episodes={len(episodes)},candidates={args.candidates},"
        f"dim={encoded.shape[-1]},raw_text=false to {args.output}"
    )


if __name__ == "__main__":
    main()
