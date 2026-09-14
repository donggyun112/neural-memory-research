from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F


def sparse_code(projected: Tensor, active: int) -> Tensor:
    """Keep the strongest `active` units and zero the rest, then renormalise.

    This is the mushroom body's arrangement: a wide random expansion whose
    activity is held sparse by reciprocal inhibition, so each event touches only
    a small fraction of the available substrate.
    """
    if projected.ndim != 2:
        raise ValueError("projected must have shape [documents, units]")
    if not 1 <= active <= projected.shape[1]:
        raise ValueError("active must lie between one and the unit count")
    kept = torch.zeros_like(projected)
    indices = projected.abs().topk(active, dim=1).indices
    kept.scatter_(1, indices, projected.gather(1, indices))
    return F.normalize(kept, dim=1)


def store_and_recall(keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    """Write every document with the delta rule, then read each one back.

    Returns two per-document scores. Fidelity is the cosine between what the
    state returns for a document's own key and what was written there.
    Discrimination is whether that read is closer to the document's own value
    than to any other stored value, which is the quantity the sparsening result
    is about: removing inhibition impairs telling similar odours apart, not
    recall strength as such. Nothing is selected and nothing is trained; the only
    thing degrading either score is interference.
    """
    if len(keys) != len(values):
        raise ValueError("keys and values must describe the same documents")
    matrix = values.new_zeros(values.shape[1], keys.shape[1])
    for key, value in zip(keys, values, strict=True):
        prediction = matrix @ key
        matrix = matrix + torch.outer(value - prediction, key)
    read = F.normalize(keys @ matrix.T, dim=1)
    normalized = F.normalize(values, dim=1)
    agreement = read @ normalized.T
    fidelity = agreement.diagonal()
    rivals = agreement.masked_fill(
        torch.eye(len(values), dtype=torch.bool, device=values.device), -torch.inf
    )
    discrimination = (fidelity > rivals.max(dim=1).values).float()
    return fidelity, discrimination


def similar_set(documents: Tensor, size: int, seed_index: int) -> Tensor:
    """The size documents most like one another, taken around a seed."""
    similarity = documents @ documents[seed_index]
    return similarity.topk(size).indices


def dissimilar_set(documents: Tensor, size: int, seed_index: int) -> Tensor:
    """A greedily spread set, each new document least like those already chosen."""
    chosen = [seed_index]
    similarity = documents @ documents.T
    while len(chosen) < size:
        worst = similarity[chosen].max(dim=0).values
        worst[torch.tensor(chosen)] = float("inf")
        chosen.append(int(worst.argmin()))
    return torch.tensor(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Does sparse expansion help, and only for similar documents?"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--dense-dim", type=int, default=32)
    parser.add_argument("--expanded-dim", type=int, default=512)
    parser.add_argument("--loads", default="2,4,8,16,32,64")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    documents = F.normalize(payload["candidates"].flatten(0, 1), dim=-1)
    feature_dim = documents.shape[-1]
    loads = [int(value) for value in args.loads.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]

    results: dict[str, dict[str, dict[str, float]]] = {}
    for regime, builder in (("similar", similar_set), ("dissimilar", dissimilar_set)):
        for arm in ("dense_small", "dense_large", "sparse_large"):
            per_load: dict[str, float] = {}
            per_load_discrimination: dict[str, float] = {}
            spread: dict[str, float] = {}
            for load in loads:
                scores: list[float] = []
                separations: list[float] = []
                for seed in seeds:
                    generator = torch.Generator().manual_seed(seed)
                    width = args.dense_dim if arm == "dense_small" else args.expanded_dim
                    key_projection = torch.randn(
                        feature_dim, width, generator=generator
                    ) / width**0.5
                    value_projection = torch.randn(
                        feature_dim, args.value_dim, generator=generator
                    ) / args.value_dim**0.5
                    for episode in range(args.episodes):
                        anchor = int(
                            torch.randint(
                                len(documents), (1,), generator=generator
                            )
                        )
                        chosen = builder(documents, load, anchor)
                        block = documents[chosen]
                        projected = torch.tanh(block @ key_projection)
                        keys = (
                            sparse_code(projected, args.dense_dim)
                            if arm == "sparse_large"
                            else F.normalize(projected, dim=1)
                        )
                        values = F.normalize(torch.tanh(block @ value_projection), dim=1)
                        fidelity, discrimination = store_and_recall(keys, values)
                        scores.append(float(fidelity.mean()))
                        separations.append(float(discrimination.mean()))
                per_load[str(load)] = sum(scores) / len(scores)
                per_load_discrimination[str(load)] = sum(separations) / len(separations)
                centre = per_load[str(load)]
                spread[str(load)] = (
                    sum((score - centre) ** 2 for score in scores) / len(scores)
                ) ** 0.5
            results[f"{regime}/{arm}"] = {
                "fidelity": per_load,
                "discrimination": per_load_discrimination,
                "std": spread,
            }

    output = {
        "config": vars(args),
        "documents": len(documents),
        "active_units_per_write": args.dense_dim,
        "substrate": {
            "dense_small": args.dense_dim * args.value_dim,
            "dense_large": args.expanded_dim * args.value_dim,
            "sparse_large": args.expanded_dim * args.value_dim,
        },
        "plasticity_per_write": {
            "dense_small": args.dense_dim * args.value_dim,
            "dense_large": args.expanded_dim * args.value_dim,
            "sparse_large": args.dense_dim * args.value_dim,
        },
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
