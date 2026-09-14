from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch import Tensor
from torch.nn import functional as F

from neural_memory.trainable import TrainableMemory


def hit_rate(scores: Tensor, targets: Tensor) -> float:
    """Fraction of questions whose highest-scoring session is the evidence one."""
    return float((scores.argmax(dim=-1) == targets).float().mean())


def hit_rate_at(scores: Tensor, targets: Tensor, keep: int) -> float:
    """Fraction of questions whose evidence session survives into the top ``keep``."""
    kept = scores.topk(keep, dim=-1).indices
    return float((kept == targets.unsqueeze(-1)).any(dim=-1).float().mean())


@torch.no_grad()
def cosine_scores(candidates: Tensor, queries: Tensor) -> Tensor:
    """The retrieval the encoder already supports without any memory at all."""
    return torch.einsum('end,ed->en', candidates, queries)


@torch.no_grad()
def projected_scores(model: TrainableMemory, candidates: Tensor, queries: Tensor) -> Tensor:
    """Cosine in the trained key space, with nothing written down.

    Training moves the projection as well as the store, so without this control a
    gain from the projection alone would be read as a gain from the memory.
    """
    keys = F.normalize(torch.tanh(model.key_projection(candidates)), dim=-1)
    cues = F.normalize(torch.tanh(model.key_projection(queries)), dim=-1)
    return torch.einsum('end,ed->en', keys, cues)


@torch.no_grad()
def memory_scores(
    model: TrainableMemory, candidates: Tensor, queries: Tensor, weak_first: float
) -> Tensor:
    model.eval()
    return torch.stack(
        [
            model.probe(block, cue.unsqueeze(0), weak_first)[0]
            for block, cue in zip(candidates, queries, strict=True)
        ]
    )


@torch.no_grad()
def code_overlap(
    model: TrainableMemory, candidates: Tensor, queries: Tensor, targets: Tensor, width: int
) -> float:
    """Fraction of a question's active units that its evidence session also uses.

    A sparse key addresses the store through the units it switches on, so a cue
    that shares none of them cannot reach what was written, however well the
    write itself discriminated.
    """
    keys = torch.tanh(model.key_projection(candidates)).abs().topk(width, dim=-1).indices
    cues = torch.tanh(model.key_projection(queries)).abs().topk(width, dim=-1).indices
    evidence = keys[torch.arange(len(targets)), targets]
    shared = [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(evidence, cues, strict=True)]
    return sum(shared) / (len(shared) * width)


def with_distractors(candidates: Tensor, extra: int, generator: torch.Generator) -> Tensor:
    """Pad every question's candidate set with sessions belonging to other questions.

    The benchmark hands each question eight sessions, which any linear scan can
    hold. A compressive store only has something to sell once the set is larger
    than that, so the load is raised here rather than assumed away.
    """
    if extra < 1:
        return candidates
    episodes, traces, features = candidates.shape
    pool = candidates.reshape(-1, features)
    drawn = torch.randint(len(pool), (episodes, extra), generator=generator)
    own = torch.arange(episodes).unsqueeze(-1) * traces
    # Redraw anything that landed on the question's own sessions, which would
    # otherwise duplicate the evidence and flatter the score.
    clash = (drawn >= own) & (drawn < own + traces)
    drawn[clash] = (drawn[clash] + traces) % len(pool)
    return torch.cat([candidates, pool[drawn]], dim=1)


def report(scores: Tensor, targets: Tensor, keep: int) -> dict[str, float]:
    return {
        "hit_rate": hit_rate(scores, targets),
        f"hit_rate_at_{keep}": hit_rate_at(scores, targets, keep),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Does the store surface the evidence session for the real question?"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--key-dim", type=int, default=512)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--keep", type=int, default=2, help="slots the generator gets")
    parser.add_argument("--full-components", action="store_true")
    parser.add_argument(
        "--fixed-active",
        type=int,
        default=0,
        help="hold the code width instead of letting it follow the state; 0 keeps the adaptive rule",
    )
    parser.add_argument("--weak-first", type=float, default=1.0)
    parser.add_argument(
        "--distractors",
        type=int,
        default=0,
        help="extra sessions drawn from other questions, to raise the load past the eight candidates",
    )
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    candidates = F.normalize(payload["candidates"], dim=-1)
    queries = F.normalize(payload["query"], dim=-1)
    targets = payload["targets"].long()
    episodes, traces, features = candidates.shape

    rows: list[dict[str, float]] = []
    for seed in (int(value) for value in args.seeds.split(",")):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(episodes, generator=generator)
        cut = int(episodes * (1.0 - args.holdout))
        train_index, eval_index = order[:cut], order[cut:]
        loaded = with_distractors(candidates, args.distractors, generator)

        model = TrainableMemory(
            features,
            key_dim=args.key_dim,
            value_dim=args.value_dim,
            full_components=args.full_components,
            adaptive=not args.fixed_active,
            fixed_active=args.fixed_active or 32,
        )
        held_candidates = loaded[eval_index]
        held_queries = queries[eval_index]
        held_targets = targets[eval_index]

        row = {"seed": float(seed)}
        for name, scores in (
            ("cosine", cosine_scores(held_candidates, held_queries)),
            (
                "untrained_projection",
                projected_scores(model, held_candidates, held_queries),
            ),
            (
                "untrained_memory",
                memory_scores(model, held_candidates, held_queries, args.weak_first),
            ),
        ):
            row.update({f"{name}_{key}": value for key, value in report(scores, held_targets, args.keep).items()})

        optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        model.train()
        for step in range(args.steps):
            pick = int(train_index[step % len(train_index)])
            logits = model.probe(
                loaded[pick], queries[pick].unsqueeze(0), args.weak_first
            )
            loss = F.cross_entropy(logits, targets[pick].unsqueeze(0))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

        for name, scores in (
            ("trained_projection", projected_scores(model, held_candidates, held_queries)),
            (
                "trained_memory",
                memory_scores(model, held_candidates, held_queries, args.weak_first),
            ),
        ):
            row.update({f"{name}_{key}": value for key, value in report(scores, held_targets, args.keep).items()})
        row["code_overlap"] = code_overlap(
            model, held_candidates, held_queries, held_targets, args.fixed_active or 32
        )
        rows.append(row)

    output = {
        "config": vars(args),
        "episodes": episodes,
        "traces": traces + args.distractors,
        "chance": 1.0 / (traces + args.distractors),
        "held_out": {
            metric: {
                "mean": mean(row[metric] for row in rows),
                "std": pstdev(row[metric] for row in rows),
            }
            for metric in rows[0]
            if metric != "seed"
        },
        "per_seed": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
