from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from analyze_sparse_capacity import dissimilar_set, similar_set, sparse_code


def store(
    keys: Tensor,
    values: Tensor,
    *,
    decay: float,
    adaptive: bool,
    sensitivity: float,
) -> Tensor:
    """Delta-rule storage whose forgetting can respond to local crowding.

    A constant decay erodes everything at the same rate whether or not it is
    under pressure, which Phase 21 found never helps. An adaptive rule instead
    forgets where the interference actually is: each key unit keeps a running
    count of how much has been written onto it, and the columns of the state
    belonging to crowded units decay faster than the rest. That is the form the
    claim about decay adapting to interference actually takes, and a sweep over
    constants cannot test it.
    """
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be inside (0, 1]")
    if sensitivity < 0.0:
        raise ValueError("sensitivity cannot be negative")
    matrix = values.new_zeros(values.shape[1], keys.shape[1])
    crowding = torch.zeros(keys.shape[1])
    for key, value in zip(keys, values, strict=True):
        if adaptive:
            # Units carrying more history forget faster; quiet ones are left alone.
            per_unit = torch.exp(-sensitivity * crowding)
            matrix = matrix * per_unit[None, :]
        else:
            matrix = decay * matrix
        matrix = matrix + torch.outer(value - matrix @ key, key)
        crowding = crowding + key.abs()
    return matrix


def fidelity(matrix: Tensor, keys: Tensor, values: Tensor) -> float:
    read = F.normalize(keys @ matrix.T, dim=1)
    return float((read * F.normalize(values, dim=1)).sum(dim=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Does a decay that responds to crowding beat the best constant?"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--key-dim", type=int, default=512)
    parser.add_argument("--active", type=int, default=32)
    parser.add_argument("--loads", default="8,16,32,64,128")
    parser.add_argument("--decays", default="1.0,0.99,0.95")
    parser.add_argument("--sensitivities", default="0.001,0.01,0.05")
    parser.add_argument("--regime", choices=("similar", "dissimilar"), default="similar")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    documents = F.normalize(payload["candidates"].flatten(0, 1), dim=-1)
    feature_dim = documents.shape[-1]
    builder = similar_set if args.regime == "similar" else dissimilar_set
    loads = [int(value) for value in args.loads.split(",")]
    decays = [float(value) for value in args.decays.split(",")]
    sensitivities = [float(value) for value in args.sensitivities.split(",")]

    arms = [(f"constant_{d:g}", d, False, 0.0) for d in decays]
    arms += [(f"adaptive_{s:g}", 1.0, True, s) for s in sensitivities]

    by_load: dict[str, dict[str, float]] = {}
    for load in loads:
        scores: dict[str, list[float]] = {name: [] for name, *_ in arms}
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
                keys = sparse_code(torch.tanh(block @ key_projection), args.active)
                values = F.normalize(torch.tanh(block @ value_projection), dim=1)
                for name, decay, adaptive, sensitivity in arms:
                    scores[name].append(
                        fidelity(
                            store(
                                keys,
                                values,
                                decay=decay,
                                adaptive=adaptive,
                                sensitivity=sensitivity,
                            ),
                            keys,
                            values,
                        )
                    )
        by_load[str(load)] = {
            name: sum(values) / len(values) for name, values in scores.items()
        }
        best_constant = max(
            (name for name, *_ in arms if name.startswith("constant")),
            key=lambda name: by_load[str(load)][name],
        )
        best_adaptive = max(
            (name for name, *_ in arms if name.startswith("adaptive")),
            key=lambda name: by_load[str(load)][name],
        )
        by_load[str(load)]["best_constant"] = by_load[str(load)][best_constant]
        by_load[str(load)]["best_adaptive"] = by_load[str(load)][best_adaptive]
        by_load[str(load)]["adaptive_margin"] = (
            by_load[str(load)][best_adaptive] - by_load[str(load)][best_constant]
        )

    output = {"config": vars(args), "documents": len(documents), "by_load": by_load}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
