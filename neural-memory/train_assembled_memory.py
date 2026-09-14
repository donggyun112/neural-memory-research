from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch import Tensor
from torch.nn import functional as F

from analyze_sparse_capacity import dissimilar_set, similar_set
from neural_memory.trainable import TrainableMemory


def episode_blocks(
    documents: Tensor, pool: Tensor, load: int, count: int, builder, generator: torch.Generator
) -> list[Tensor]:
    """Draw episodes whose documents come only from the given pool."""
    subset = documents[pool]
    return [
        subset[builder(subset, load, int(torch.randint(len(subset), (1,), generator=generator)))]
        for _ in range(count)
    ]


def evaluate(model: TrainableMemory, blocks: list[Tensor], weak_first: float = 1.0) -> float:
    model.eval()
    return mean(model.discrimination(block, weak_first) for block in blocks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the assembled store's projections against discrimination"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--key-dim", type=int, default=512)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--load", type=int, default=32)
    parser.add_argument("--regime", choices=("similar", "dissimilar"), default="similar")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--fixed-sparsity", action="store_true")
    parser.add_argument("--learned-width", action="store_true")
    parser.add_argument("--sampled-width", action="store_true")
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--full-components", action="store_true")
    parser.add_argument("--weak-first", type=float, default=0.4)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    documents = F.normalize(payload["candidates"].flatten(0, 1), dim=-1)
    builder = similar_set if args.regime == "similar" else dissimilar_set

    rows: list[dict[str, float]] = []
    for seed in (int(value) for value in args.seeds.split(",")):
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(documents), generator=generator)
        cut = int(len(documents) * (1.0 - args.holdout))
        train_pool, eval_pool = order[:cut], order[cut:]
        # Episodes are drawn from disjoint document pools, so a projection that
        # memorised particular documents would show it here.
        eval_blocks = episode_blocks(
            documents, eval_pool, args.load, args.eval_episodes, builder, generator
        )
        seen_blocks = episode_blocks(
            documents, train_pool, args.load, args.eval_episodes, builder, generator
        )

        model = TrainableMemory(
            documents.shape[-1],
            key_dim=args.key_dim,
            value_dim=args.value_dim,
            adaptive=not args.fixed_sparsity,
            learned_width=args.learned_width,
            sampled_width=args.sampled_width,
            full_components=args.full_components,
        )
        before = evaluate(model, eval_blocks, args.weak_first)
        before_seen = evaluate(model, seen_blocks, args.weak_first)
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        model.train()
        baseline: float | None = None
        for _ in range(args.steps):
            block = episode_blocks(
                documents, train_pool, args.load, 1, builder, generator
            )[0]
            logits = model(block, weak_first=args.weak_first)
            loss = F.cross_entropy(logits, torch.arange(len(block)))
            total = loss
            if args.sampled_width and model._last_log_probabilities:
                # The width is sampled, so its parameters get a policy
                # gradient against the episode's own loss rather than a
                # relaxation that would soften the selection.
                reward = -float(loss.detach())
                baseline = reward if baseline is None else 0.9 * baseline + 0.1 * reward
                advantage = reward - baseline
                total = total - args.policy_weight * advantage * torch.stack(
                    model._last_log_probabilities
                ).sum()
            optimiser.zero_grad(set_to_none=True)
            total.backward()
            optimiser.step()
        after = evaluate(model, eval_blocks, args.weak_first)
        after_seen = evaluate(model, seen_blocks, args.weak_first)
        rows.append(
            {
                "untrained": before,
                "trained": after,
                "gain": after - before,
                "untrained_seen": before_seen,
                "trained_seen": after_seen,
                "gain_seen": after_seen - before_seen,
                "mean_width": sum(model._last_widths) / len(model._last_widths),
                "policy_base": float(model.width_policy.log_base.exp()),
                "policy_exponent": float(model.width_policy.exponent),
                "slow_strength": float(model.slow_logit.sigmoid()),
                "slow_decay": float(model.slow_decay_logit.sigmoid()),
                "tag_decay": float(model.tag_decay_logit.sigmoid()),
                "capture": float(model.capture_logit.sigmoid()),
            }
        )

    output = {
        "config": vars(args),
        "documents": len(documents),
        "held_out": {
            metric: {
                "mean": mean(row[metric] for row in rows),
                "std": pstdev(row[metric] for row in rows),
            }
            for metric in rows[0]
        },
        "per_seed": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
