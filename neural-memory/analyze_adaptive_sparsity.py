from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from analyze_sparse_capacity import dissimilar_set, similar_set, sparse_code, store_and_recall


def crowding_signal(matrix: Tensor, projected: Tensor) -> float:
    """How much the state already answers a probe it has never been given.

    Available before the active count is chosen, and zero on an empty state, but
    it saturates badly: over a hundred-and-twenty-eight-fold change in load it
    only doubles, from 0.45 to 1.01, so no rule reading it can span the range of
    active counts the sweep says is needed.
    """
    return float((matrix @ F.normalize(projected, dim=0)).norm())


def state_size(matrix: Tensor) -> float:
    """The size of the state itself, which tracks load without saturating.

    It runs 1.00, 1.98, 3.01, 5.31 at loads of one, eight, thirty-two and a
    hundred and twenty-eight, and the active counts the sweep prefers at those
    loads sit close to a fixed power of it. Entirely internal: the memory is
    reading its own magnitude, not being told how much it holds.
    """
    return float(matrix.norm())


def adaptive_store(
    projected: Tensor,
    values: Tensor,
    *,
    base: float,
    exponent: float,
    lowest: int,
    highest: int,
) -> tuple[Tensor, Tensor, list[int]]:
    """Choose each document's active count from how large the state has grown."""
    if not 1 <= lowest <= highest <= projected.shape[1]:
        raise ValueError("active counts must satisfy 1 <= lowest <= highest <= units")
    if base <= 0.0:
        raise ValueError("base must be positive")
    matrix = values.new_zeros(values.shape[1], projected.shape[1])
    keys = []
    counts: list[int] = []
    for row, value in zip(projected, values, strict=True):
        size = max(state_size(matrix), 1.0)
        active = int(min(highest, max(lowest, round(base * size**exponent))))
        key = sparse_code(row[None], active)[0]
        keys.append(key)
        counts.append(active)
        matrix = matrix + torch.outer(value - matrix @ key, key)
    return torch.stack(keys), matrix, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Can the store infer its own load and set its sparsity from it?"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--key-dim", type=int, default=512)
    parser.add_argument("--fixed", type=int, default=32)
    parser.add_argument("--base", type=float, default=6.1)
    parser.add_argument("--exponent", type=float, default=1.4)
    parser.add_argument("--lowest", type=int, default=8)
    parser.add_argument("--highest", type=int, default=96)
    parser.add_argument("--candidates", default="8,16,32,64,128")
    parser.add_argument("--loads", default="8,32,128")
    parser.add_argument("--regime", choices=("similar", "dissimilar"), default="similar")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    documents = F.normalize(payload["candidates"].flatten(0, 1), dim=-1)
    feature_dim = documents.shape[-1]
    builder = similar_set if args.regime == "similar" else dissimilar_set
    loads = [int(value) for value in args.loads.split(",")]
    candidates = [int(value) for value in args.candidates.split(",")]

    by_load: dict[str, dict[str, float]] = {}
    for load in loads:
        fixed: list[float] = []
        adaptive: list[float] = []
        chosen_counts: list[float] = []
        oracle: dict[int, list[float]] = {count: [] for count in candidates}
        for seed in (int(value) for value in args.seeds.split(",")):
            generator = torch.Generator().manual_seed(seed)
            key_projection = torch.randn(
                feature_dim, args.key_dim, generator=generator
            ) / args.key_dim**0.5
            value_projection = torch.randn(
                feature_dim, args.value_dim, generator=generator
            ) / args.value_dim**0.5
            for _ in range(args.episodes):
                anchor = int(torch.randint(len(documents), (1,), generator=generator))
                block = documents[builder(documents, load, anchor)]
                projected = torch.tanh(block @ key_projection)
                values = F.normalize(torch.tanh(block @ value_projection), dim=1)
                for count in candidates:
                    keys = sparse_code(projected, count)
                    score = float(store_and_recall(keys, values)[1].mean())
                    oracle[count].append(score)
                    if count == args.fixed:
                        fixed.append(score)
                keys, _, counts = adaptive_store(
                    projected,
                    values,
                    base=args.base,
                    exponent=args.exponent,
                    lowest=args.lowest,
                    highest=args.highest,
                )
                adaptive.append(float(store_and_recall(keys, values)[1].mean()))
                chosen_counts.append(sum(counts) / len(counts))
        averaged = {count: sum(v) / len(v) for count, v in oracle.items()}
        best = max(averaged, key=lambda count: averaged[count])
        by_load[str(load)] = {
            "fixed": sum(fixed) / len(fixed),
            "adaptive": sum(adaptive) / len(adaptive),
            "oracle_best": averaged[best],
            "oracle_active": float(best),
            "mean_chosen_active": sum(chosen_counts) / len(chosen_counts),
            "adaptive_over_fixed": sum(adaptive) / len(adaptive) - sum(fixed) / len(fixed),
            "adaptive_under_oracle": sum(adaptive) / len(adaptive) - averaged[best],
        }

    output = {"config": vars(args), "documents": len(documents), "by_load": by_load}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
