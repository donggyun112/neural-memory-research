from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from train_deferred_consolidation import stratified_folds


def train_scorer(
    features: Tensor,
    targets: Tensor,
    *,
    hidden: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> nn.Module:
    """Fit an unconstrained listwise scorer on the raw encoder features.

    The write gate sees no query and no later event, so this bounds what any
    model could extract from a candidate's own content under the same teacher.
    """
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(features.shape[-1], hidden), nn.GELU(), nn.Linear(hidden, 1)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed + 1)
    model.train()
    for _ in range(steps):
        rows = torch.randint(len(targets), (batch_size,), generator=generator)
        logits = model(features[rows]).squeeze(-1)
        loss = F.cross_entropy(logits, targets[rows])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="How much write-time signal exists at all?"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--types", default="multi-session,temporal-reasoning")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    names = payload["question_type_names"]
    wanted = [name.strip() for name in args.types.split(",") if name.strip()]
    keep_rows = torch.zeros(len(payload["targets"]), dtype=torch.bool)
    for name in wanted:
        keep_rows |= payload["question_type"] == names.index(name)
    candidates = payload["candidates"][keep_rows]
    targets = payload["targets"][keep_rows]
    lengths = payload["lengths"][keep_rows]
    types = payload["question_type"][keep_rows]

    episodes, traces, _ = candidates.shape
    gold = targets[:, None]
    longest = float((lengths.topk(args.keep, dim=1).indices == gold).any(dim=1).float().mean())
    assignment = stratified_folds(types, args.folds)

    rows: list[float] = []
    for seed in (int(value) for value in args.seeds.split(",")):
        kept = torch.zeros(episodes, dtype=torch.bool)
        for fold in range(args.folds):
            evaluate = assignment == fold
            model = train_scorer(
                candidates[~evaluate],
                targets[~evaluate],
                hidden=args.hidden,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=seed,
            )
            model.eval()
            with torch.no_grad():
                scores = model(candidates[evaluate]).squeeze(-1)
            chosen = scores.topk(args.keep, dim=1).indices
            kept[evaluate] = (chosen == targets[evaluate][:, None]).any(dim=1)
        rows.append(float(kept.float().mean()))

    output = {
        "episodes": episodes,
        "traces": traces,
        "keep": args.keep,
        "parameters": (candidates.shape[-1] + 1) * args.hidden + args.hidden + 1,
        "random_capacity": args.keep / traces,
        "longest_k_retention": longest,
        "unconstrained_scorer_retention": {
            "mean": sum(rows) / len(rows),
            "per_seed": rows,
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
