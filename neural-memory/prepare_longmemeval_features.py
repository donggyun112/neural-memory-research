from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_memory.longmemeval import iter_longmemeval_revisits


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LongMemEval write/recall features")
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/longmemeval_s_cleaned.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/longmemeval_bge.pt")
    )
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    examples = list(
        iter_longmemeval_revisits(args.input, candidates=args.candidates)
    )
    texts = [text for example in examples for text in (*example.candidates, example.question)]
    encoder = SentenceTransformer(args.model, device=args.device)
    encoded = encoder.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).cpu()
    rows: dict[str, dict[str, list[torch.Tensor | int]]] = {
        split: {"candidates": [], "feedback": [], "targets": [], "delays": []}
        for split in ("train", "eval")
    }
    stride = args.candidates + 1
    for index, example in enumerate(examples):
        base = index * stride
        row = rows[example.split]
        row["candidates"].append(encoded[base : base + args.candidates])
        row["feedback"].append(encoded[base + args.candidates])
        row["targets"].append(example.target_offset)
        row["delays"].append(example.delay)
    output: dict[str, torch.Tensor | str] = {
        "source": "LongMemEval-S cleaned",
        "model": args.model,
    }
    for split in ("train", "eval"):
        output[f"{split}_candidates"] = torch.stack(rows[split]["candidates"])  # type: ignore[arg-type]
        output[f"{split}_feedback"] = torch.stack(rows[split]["feedback"])  # type: ignore[arg-type]
        output[f"{split}_targets"] = torch.tensor(rows[split]["targets"], dtype=torch.long)
        output[f"{split}_delays"] = torch.tensor(rows[split]["delays"], dtype=torch.long)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"saved train={len(rows['train']['targets'])},"
        f"eval={len(rows['eval']['targets'])},candidates={args.candidates},"
        f"dim={encoded.shape[-1]},raw_text=false to {args.output}"
    )


if __name__ == "__main__":
    main()
