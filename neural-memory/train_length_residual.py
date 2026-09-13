from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch import Tensor
from torch.nn import functional as F

from analyze_length_baseline import claude_slot_lengths, longmemeval_slot_lengths
from neural_memory.global_feedback import load_global_feedback_banks
from train_revisit_write_recall import evaluate, train_model


def length_features(lengths: Tensor, log_mean: float, log_std: float) -> Tensor:
    """Turn raw candidate lengths into the two cues a length rule could exploit.

    The first channel is the standardized log length and the second is the
    within-episode rank, which is exactly what a "keep the longest" rule reads.
    """
    if lengths.ndim != 2:
        raise ValueError("lengths must have shape [episodes, traces]")
    if log_std <= 0:
        raise ValueError("log_std must be positive")
    traces = lengths.shape[1]
    if traces < 2:
        raise ValueError("at least two traces are required to rank lengths")
    standardized = (torch.log1p(lengths) - log_mean) / log_std
    rank = lengths.argsort(dim=1).argsort(dim=1).to(lengths.dtype) / (traces - 1)
    return torch.stack((standardized, rank), dim=-1)


def build_variant(
    name: str, embeddings: Tensor, feedback: Tensor, lengths: Tensor
) -> tuple[Tensor, Tensor]:
    """Assemble the candidate and recall-context tensors for one ablation arm."""
    if name == "embedding":
        return embeddings, feedback
    if name == "length":
        return lengths, torch.zeros(len(feedback), lengths.shape[-1])
    if name == "embedding_length":
        return torch.cat((embeddings, lengths), dim=-1), F.pad(
            feedback, (0, lengths.shape[-1])
        )
    raise ValueError(f"unknown variant: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Does the writer still help once candidate length is explicit?"
    )
    parser.add_argument("--source", choices=("claude", "longmemeval"), default="claude")
    parser.add_argument("--features", type=Path, default=Path("artifacts/claude_global_feedback.pt"))
    parser.add_argument("--history", type=Path, default=Path.home() / ".claude" / "history.jsonl")
    parser.add_argument("--input", type=Path, default=Path("artifacts/longmemeval_s_cleaned.json"))
    parser.add_argument(
        "--variants", default="embedding,length,embedding_length"
    )
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--write-steps", type=int, default=1000)
    parser.add_argument("--recall-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--keep-ratio", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    banks = load_global_feedback_banks(args.features)
    traces = banks.train_candidates.shape[1]
    if args.source == "claude":
        splits = claude_slot_lengths(args.history, traces)
    else:
        splits = longmemeval_slot_lengths(args.input, traces)
    lengths = {split: value[0] for split, value in splits.items()}
    for split, expected in (("train", banks.train_targets), ("eval", banks.eval_targets)):
        if not torch.equal(splits[split][1], expected):
            raise RuntimeError(
                f"recomputed {split} lengths do not align with the feature artifact; "
                "regenerate the embeddings from the same source"
            )
    train_log = torch.log1p(lengths["train"])
    log_mean, log_std = float(train_log.mean()), float(train_log.std())
    length_banks = {
        split: length_features(value, log_mean, log_std) for split, value in lengths.items()
    }

    keep = max(1, int(round(args.keep_ratio * traces)))
    heuristic = float(
        (
            lengths["eval"].topk(keep, dim=1).indices == banks.eval_targets[:, None]
        ).any(dim=1).float().mean()
    )

    seeds = [int(value) for value in args.seeds.split(",")]
    variants: dict[str, dict[str, dict[str, float]]] = {}
    for name in args.variants.split(","):
        train_candidates, train_feedback = build_variant(
            name, banks.train_candidates, banks.train_feedback, length_banks["train"]
        )
        eval_candidates, eval_feedback = build_variant(
            name, banks.eval_candidates, banks.eval_feedback, length_banks["eval"]
        )
        rows: list[dict[str, float]] = []
        for seed in seeds:
            model = train_model(
                train_candidates,
                train_feedback,
                banks.train_targets,
                keep_ratio=args.keep_ratio,
                write_steps=args.write_steps,
                recall_steps=args.recall_steps,
                batch_size=args.batch_size,
                memory_dim=args.memory_dim,
                learning_rate=args.learning_rate,
                seed=seed,
                shuffle_write_teacher=False,
                device=torch.device(args.device),
            )
            rows.append(
                asdict(
                    evaluate(
                        model,
                        eval_candidates,
                        eval_feedback,
                        banks.eval_targets,
                        device=torch.device(args.device),
                    )
                )
            )
        variants[name] = {
            metric: {
                "mean": mean(row[metric] for row in rows),
                "std": pstdev(row[metric] for row in rows),
            }
            for metric in rows[0]
        }

    output = {
        "config": vars(args),
        "source": args.source,
        "traces": traces,
        "kept_slots": keep,
        "eval_examples": len(banks.eval_targets),
        "longest_k_heuristic_retention": heuristic,
        "random_capacity_expectation": keep / traces,
        "variants": variants,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
