from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from analyze_opponent_modulation import coincidence, recall, write_all
from analyze_sparse_capacity import sparse_code


def allocate(
    matrix: Tensor,
    tags: Tensor,
    values: Tensor,
    probe_key: Tensor,
    *,
    sign: float,
    capture: float,
    competitive: bool,
) -> Tensor:
    """Modulate the traces, optionally under a fixed total-plasticity budget.

    Competitive allocation subtracts the mean gain so the gains sum to zero,
    which makes recruitment zero-sum: whatever one trace takes has to come out of
    the others. That is the constraint the constant engram population implies,
    and it is the difference between reallocating and simply adding.
    """
    current = coincidence(matrix, tags, probe_key, values)
    weights = current.clamp(min=0.0)
    wanted = torch.full_like(current, 1.0 if sign > 0 else 0.0)
    gain = capture * weights * (wanted - current)
    if competitive:
        gain = gain - gain.mean()
    return matrix + torch.einsum("t,tij->ij", gain, tags)


def budget(matrix: Tensor, updated: Tensor) -> float:
    """Total plasticity spent, so variants can be compared at equal cost."""
    return float((updated - matrix).norm())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Does a fixed allocation budget buy selectivity?"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--key-dim", type=int, default=512)
    parser.add_argument("--active", type=int, default=32)
    parser.add_argument("--load", type=int, default=16)
    parser.add_argument("--capture", type=float, default=1.0)
    parser.add_argument("--sign", type=float, default=-1.0)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    documents = F.normalize(payload["candidates"].flatten(0, 1), dim=-1)
    feature_dim = documents.shape[-1]

    collected: dict[str, list[float]] = {
        name: []
        for name in (
            "free_target",
            "free_others",
            "free_selectivity",
            "free_total_recall_shift",
            "competitive_target",
            "competitive_others",
            "competitive_selectivity",
            "competitive_total_recall_shift",
            "matched_free_target",
            "matched_competitive_target",
        )
    }

    for seed in (int(value) for value in args.seeds.split(",")):
        generator = torch.Generator().manual_seed(seed)
        key_projection = torch.randn(
            feature_dim, args.key_dim, generator=generator
        ) / args.key_dim**0.5
        value_projection = torch.randn(
            feature_dim, args.value_dim, generator=generator
        ) / args.value_dim**0.5
        for _ in range(args.episodes):
            chosen = torch.randperm(len(documents), generator=generator)[: args.load]
            block = documents[chosen]
            keys = sparse_code(torch.tanh(block @ key_projection), args.active)
            values = F.normalize(torch.tanh(block @ value_projection), dim=1)
            matrix, tags = write_all(keys, values)
            before = recall(matrix, keys, values)
            target = int(torch.randint(args.load, (1,), generator=generator))
            others = [index for index in range(args.load) if index != target]

            spent: dict[str, float] = {}
            for label, competitive in (("free", False), ("competitive", True)):
                updated = allocate(
                    matrix,
                    tags,
                    values,
                    keys[target],
                    sign=args.sign,
                    capture=args.capture,
                    competitive=competitive,
                )
                spent[label] = budget(matrix, updated)
                after = recall(updated, keys, values)
                moved = float(abs(after[target] - before[target])) + float(
                    (after[others] - before[others]).abs().sum()
                )
                collected[f"{label}_target"].append(float(after[target] - before[target]))
                collected[f"{label}_others"].append(
                    float((after[others] - before[others]).mean())
                )
                collected[f"{label}_selectivity"].append(
                    float(abs(after[target] - before[target])) / moved if moved > 1e-9 else 0.0
                )
                collected[f"{label}_total_recall_shift"].append(
                    float(after.sum() - before.sum())
                )

            # Equal-cost comparison: rescale each arm so both spend the same
            # plasticity, since a constraint that simply spends less is not a
            # fair win and one that spends more is not a real one either.
            reference = min(spent.values())
            for label, competitive in (("free", False), ("competitive", True)):
                if spent[label] <= 1e-9:
                    collected[f"matched_{label}_target"].append(0.0)
                    continue
                updated = allocate(
                    matrix,
                    tags,
                    values,
                    keys[target],
                    sign=args.sign,
                    capture=args.capture * reference / spent[label],
                    competitive=competitive,
                )
                after = recall(updated, keys, values)
                collected[f"matched_{label}_target"].append(
                    float(after[target] - before[target])
                )

    output = {
        "config": vars(args),
        "documents": len(documents),
        "measurements": {
            name: sum(values) / len(values) for name, values in collected.items()
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
