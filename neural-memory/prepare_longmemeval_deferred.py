from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_memory.longmemeval import iter_longmemeval_deferred


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deferred-consolidation features from multi-evidence questions"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/longmemeval_s_cleaned.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    episodes = list(iter_longmemeval_deferred(args.input, candidates=args.candidates))
    if not episodes:
        raise RuntimeError("no multi-evidence episodes were produced")
    texts = [
        text
        for episode in episodes
        for text in (*episode.candidates, episode.consolidation, episode.question)
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
    candidates = torch.stack(
        [encoded[index * stride : index * stride + args.candidates] for index in range(len(episodes))]
    )
    consolidation = torch.stack(
        [encoded[index * stride + args.candidates] for index in range(len(episodes))]
    )
    query = torch.stack(
        [encoded[index * stride + args.candidates + 1] for index in range(len(episodes))]
    )
    types = sorted({episode.question_type for episode in episodes})
    output = {
        "source": "LongMemEval-S cleaned, multi-evidence",
        "model": args.model,
        "candidates": candidates,
        "consolidation": consolidation,
        "query": query,
        "targets": torch.tensor([episode.target_offset for episode in episodes], dtype=torch.long),
        # Candidate lengths travel with the features so the training-free length
        # baseline never needs the raw corpus again.
        "lengths": torch.tensor(
            [[float(len(text)) for text in episode.candidates] for episode in episodes]
        ),
        "evidence_gap": torch.tensor(
            [episode.evidence_gap for episode in episodes], dtype=torch.long
        ),
        "question_type": torch.tensor(
            [types.index(episode.question_type) for episode in episodes], dtype=torch.long
        ),
        "question_type_names": types,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"saved episodes={len(episodes)},candidates={args.candidates},"
        f"dim={encoded.shape[-1]},types={types},raw_text=false to {args.output}"
    )


if __name__ == "__main__":
    main()
