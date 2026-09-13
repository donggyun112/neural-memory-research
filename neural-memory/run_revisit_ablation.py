from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev

import torch
from torch import Tensor

from neural_memory.global_feedback import load_global_feedback_banks
from train_revisit_write_recall import add_causal_write_features, evaluate, train_model


VARIANTS: dict[str, dict[str, object]] = {
    "staged_bce": {},
    "untrained_writer": {"write_steps": 0},
    "random_write_teacher": {"shuffle_write_teacher": True},
    "staged_listwise": {"write_objective": "listwise"},
    "causal_bce": {"causal_features": True},
    "causal_position": {"causal_features": True, "causal_feature_mode": "position"},
    "causal_novelty": {"causal_features": True, "causal_feature_mode": "novelty"},
    "causal_surprise": {"causal_features": True, "causal_feature_mode": "surprise"},
    "causal_novelty_surprise": {
        "causal_features": True,
        "causal_feature_mode": "novelty_surprise",
    },
    "causal_listwise": {
        "write_objective": "listwise",
        "causal_features": True,
    },
    "causal_listwise_joint": {
        "write_objective": "listwise",
        "causal_features": True,
        "joint_steps": 500,
    },
}
DEFAULT_VARIANTS = (
    "staged_bce,staged_listwise,causal_bce,causal_listwise,causal_listwise_joint"
)


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": mean(row[key] for row in rows),
            "std": stdev(row[key] for row in rows) if len(rows) > 1 else 0.0,
        }
        for key in rows[0]
    }


def causal_heuristics(
    candidates: Tensor, feedback: Tensor, targets: Tensor, keep_ratio: float
) -> dict[str, float]:
    augmented, _ = add_causal_write_features(
        candidates, torch.zeros(len(candidates), candidates.shape[-1])
    )
    keep = max(1, round(candidates.shape[1] * keep_ratio))
    target = targets[:, None]
    position_selected = augmented[..., -3].topk(keep, dim=1).indices
    novelty_selected = augmented[..., -2].topk(keep, dim=1).indices
    surprise_selected = augmented[..., -1].topk(keep, dim=1).indices

    def retained(indices: Tensor) -> float:
        return float((indices == target).any(dim=1).float().mean())

    def macro_retained(indices: Tensor) -> float:
        hits = (indices == target).any(dim=1)
        return mean(
            float(hits[targets == position].float().mean())
            for position in range(candidates.shape[1])
            if bool((targets == position).any())
        )

    query_scores = (
        torch.nn.functional.normalize(candidates, dim=-1)
        * torch.nn.functional.normalize(feedback, dim=-1)[:, None, :]
    ).sum(dim=-1)

    def recalled(indices: Tensor) -> float:
        allowed = torch.zeros_like(query_scores, dtype=torch.bool)
        allowed.scatter_(1, indices, True)
        predicted = query_scores.masked_fill(~allowed, -torch.inf).argmax(dim=1)
        return float((predicted == targets).float().mean())

    return {
        "random_expected": keep / candidates.shape[1],
        "most_recent": retained(position_selected),
        "most_recent_position_macro": macro_retained(position_selected),
        "most_recent_bge_recall_top1": recalled(position_selected),
        "causal_novelty": retained(novelty_selected),
        "causal_novelty_position_macro": macro_retained(novelty_selected),
        "causal_novelty_bge_recall_top1": recalled(novelty_selected),
        "causal_surprise": retained(surprise_selected),
        "causal_surprise_position_macro": macro_retained(surprise_selected),
        "causal_surprise_bge_recall_top1": recalled(surprise_selected),
    }


def query_retrieval_baselines(
    candidates: Tensor, feedback: Tensor, targets: Tensor, keep_ratio: float
) -> dict[str, float]:
    scores = (
        torch.nn.functional.normalize(candidates, dim=-1)
        * torch.nn.functional.normalize(feedback, dim=-1)[:, None, :]
    ).sum(dim=-1)
    ordering = scores.argsort(dim=1, descending=True)
    ranks = (ordering == targets[:, None]).nonzero()[:, 1] + 1
    keep = max(1, round(candidates.shape[1] * keep_ratio))
    return {
        "full_cosine_top1": float((ranks == 1).float().mean()),
        "full_cosine_top2": float((ranks <= 2).float().mean()),
        "full_cosine_mrr": float(ranks.float().reciprocal().mean()),
        "query_visible_capacity_retention": float((ranks <= keep).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed natural-revisit ablations")
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/claude_global_feedback.pt")
    )
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--keep-ratio", type=float, default=0.25)
    parser.add_argument("--write-steps", type=int, default=1000)
    parser.add_argument("--recall-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--joint-learning-rate", type=float, default=2e-4)
    parser.add_argument("--joint-teacher-weight", type=float, default=0.25)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    variants = args.variants.split(",")
    unknown = set(variants) - VARIANTS.keys()
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    seeds = [int(seed) for seed in args.seeds.split(",")]
    device = torch.device(args.device)
    banks = load_global_feedback_banks(args.features)
    output: dict[str, object] = {
        "config": vars(args),
        "train_examples": len(banks.train_targets),
        "eval_examples": len(banks.eval_targets),
        "write_heuristics": causal_heuristics(
            banks.eval_candidates, banks.eval_feedback, banks.eval_targets, args.keep_ratio
        ),
        "query_retrieval_baselines": query_retrieval_baselines(
            banks.eval_candidates,
            banks.eval_feedback,
            banks.eval_targets,
            args.keep_ratio,
        ),
        "variants": {},
    }
    for name in variants:
        config = VARIANTS[name]
        rows: list[dict[str, float]] = []
        for seed in seeds:
            model = train_model(
                banks.train_candidates,
                banks.train_feedback,
                banks.train_targets,
                keep_ratio=args.keep_ratio,
                write_steps=int(config.get("write_steps", args.write_steps)),
                recall_steps=args.recall_steps,
                batch_size=args.batch_size,
                memory_dim=args.memory_dim,
                learning_rate=args.learning_rate,
                seed=seed,
                shuffle_write_teacher=bool(config.get("shuffle_write_teacher", False)),
                write_objective=str(config.get("write_objective", "bce")),
                causal_features=bool(config.get("causal_features", False)),
                causal_feature_mode=str(config.get("causal_feature_mode", "all")),
                joint_steps=int(config.get("joint_steps", 0)),
                joint_learning_rate=args.joint_learning_rate,
                joint_teacher_weight=args.joint_teacher_weight,
                device=device,
            )
            correct = asdict(
                evaluate(
                    model,
                    banks.eval_candidates,
                    banks.eval_feedback,
                    banks.eval_targets,
                    causal_features=bool(config.get("causal_features", False)),
                    causal_feature_mode=str(config.get("causal_feature_mode", "all")),
                    device=device,
                )
            )
            shuffled = asdict(
                evaluate(
                    model,
                    banks.eval_candidates,
                    banks.eval_feedback.roll(1, dims=0),
                    banks.eval_targets,
                    causal_features=bool(config.get("causal_features", False)),
                    causal_feature_mode=str(config.get("causal_feature_mode", "all")),
                    device=device,
                )
            )
            row = {f"correct_{key}": value for key, value in correct.items()}
            row.update({f"shuffled_{key}": value for key, value in shuffled.items()})
            row["recall_top1_gap"] = correct["top1"] - shuffled["top1"]
            rows.append(row)
        output["variants"][name] = {
            "config": config,
            "per_seed": rows,
            "summary": summarize(rows),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    serialized = json.dumps(output, indent=2, sort_keys=True, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
        print(f"saved ablation results to {args.output}")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
