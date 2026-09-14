from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch import Tensor
from torch.nn import functional as F

from neural_memory.context_memory import AssociativeDeferredMemory
from train_deferred_consolidation import event_tensor, stratified_folds


def train_fold(
    candidates: Tensor,
    event: Tensor,
    query: Tensor,
    targets: Tensor,
    *,
    memory_dim: int,
    event_gain_bound: float,
    competitive_gain: bool,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> AssociativeDeferredMemory:
    """Train end to end from the recall loss alone.

    There is no write teacher and no selection objective, because there is no
    selection: the only thing the model can learn is how strongly to write, how
    much the later event should modulate, and how to read.
    """
    torch.manual_seed(seed)
    model = AssociativeDeferredMemory(
        candidates.shape[-1], memory_dim, event_gain_bound=event_gain_bound,
        competitive_gain=competitive_gain,
    )
    masks = torch.ones(candidates.shape[:2], dtype=torch.bool)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed + 1)
    model.train()
    for _ in range(steps):
        rows = torch.randint(len(targets), (batch_size,), generator=generator)
        logits, _ = model(candidates[rows], event[rows], query[rows], masks[rows])
        loss = F.cross_entropy(logits, targets[rows])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memory as a weight matrix, with no admission decision"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--types", default="multi-session,temporal-reasoning")
    parser.add_argument("--memory-dim", type=int, default=12)
    parser.add_argument("--event-gain-bound", type=float, default=1.0)
    parser.add_argument("--no-competitive-gain", action="store_true")
    parser.add_argument("--event-modes", default="correct,shuffled,blank")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    names = payload["question_type_names"]
    keep_rows = torch.zeros(len(payload["targets"]), dtype=torch.bool)
    for name in (value.strip() for value in args.types.split(",") if value.strip()):
        keep_rows |= payload["question_type"] == names.index(name)
    candidates = payload["candidates"][keep_rows]
    consolidation = payload["consolidation"][keep_rows]
    query = payload["query"][keep_rows]
    targets = payload["targets"][keep_rows]
    assignment = stratified_folds(payload["question_type"][keep_rows], args.folds)
    episodes, traces, _ = candidates.shape
    masks = torch.ones(episodes, traces, dtype=torch.bool)

    cosine = torch.einsum("etf,ef->et", candidates, query)
    references = {
        "random_top1": 1.0 / traces,
        "query_cosine_top1": float((cosine.argmax(dim=1) == targets).float().mean()),
        "persistent_scalars": args.memory_dim**2,
    }

    results: dict[str, dict[str, dict[str, float]]] = {}
    for mode in args.event_modes.split(","):
        event = event_tensor(consolidation, mode)
        rows: list[dict[str, float]] = []
        for seed in (int(value) for value in args.seeds.split(",")):
            correct = torch.zeros(episodes, dtype=torch.bool)
            gains = torch.zeros(episodes, traces)
            for fold in range(args.folds):
                evaluate = assignment == fold
                model = train_fold(
                    candidates[~evaluate],
                    event[~evaluate],
                    query[~evaluate],
                    targets[~evaluate],
                    memory_dim=args.memory_dim,
                    event_gain_bound=args.event_gain_bound,
                    competitive_gain=not args.no_competitive_gain,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=seed,
                )
                model.eval()
                with torch.no_grad():
                    logits, state = model(
                        candidates[evaluate],
                        event[evaluate],
                        query[evaluate],
                        masks[evaluate],
                    )
                correct[evaluate] = logits.argmax(dim=1) == targets[evaluate]
                gains[evaluate] = state.event_gains
            rows.append(
                {
                    "top1": float(correct.float().mean()),
                    "mean_event_gain": float(gains.mean()),
                    "mean_absolute_event_gain": float(gains.abs().mean()),
                }
            )
        results[mode] = {
            metric: {
                "mean": mean(row[metric] for row in rows),
                "std": pstdev(row[metric] for row in rows),
            }
            for metric in rows[0]
        }

    output = {
        "config": vars(args),
        "episodes": episodes,
        "traces": traces,
        "references": references,
        "event_modes": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
