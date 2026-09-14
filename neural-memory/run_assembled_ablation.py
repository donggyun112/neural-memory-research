from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from analyze_sparse_capacity import dissimilar_set, similar_set
from neural_memory.assembled import AssembledMemory, MemoryConfig


ABLATIONS = {
    "full": {},
    "no_adaptive_sparsity": {"adaptive_sparsity": False},
    "no_parallel_store": {"parallel_stores": False},
    "no_tags": {"tags": False},
    "with_decay": {"fast_decay": 0.95},
    "dense_code": {"adaptive_sparsity": False, "fixed_active": 128},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Which validated parts still earn their place once assembled?"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--loads", default="8,32,128")
    parser.add_argument("--regime", choices=("similar", "dissimilar"), default="similar")
    parser.add_argument("--weak-strength", type=float, default=0.4)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    documents = F.normalize(payload["candidates"].flatten(0, 1), dim=-1)
    feature_dim = documents.shape[-1]
    builder = similar_set if args.regime == "similar" else dissimilar_set
    loads = [int(value) for value in args.loads.split(",")]
    base = MemoryConfig()

    results: dict[str, dict[str, float]] = {}
    for name, overrides in ABLATIONS.items():
        config = replace(base, **overrides)
        memory = AssembledMemory(config)
        scores: dict[int, list[float]] = {load: [] for load in loads}
        counts: dict[int, list[float]] = {load: [] for load in loads}
        for seed in (int(value) for value in args.seeds.split(",")):
            generator = torch.Generator().manual_seed(seed)
            key_projection = torch.randn(
                feature_dim, config.key_dim, generator=generator
            ) / config.key_dim**0.5
            value_projection = torch.randn(
                feature_dim, config.value_dim, generator=generator
            ) / config.value_dim**0.5
            for load in loads:
                for _ in range(args.episodes):
                    anchor = int(
                        torch.randint(len(documents), (1,), generator=generator)
                    )
                    block = documents[builder(documents, load, anchor)]
                    projected = torch.tanh(block @ key_projection)
                    values = F.normalize(torch.tanh(block @ value_projection), dim=1)
                    state = memory.empty()
                    keys = []
                    for index, (row, value) in enumerate(
                        zip(projected, values, strict=True)
                    ):
                        # One document in each episode arrives weakly, so the
                        # tags and the later event have something to act on.
                        strength = args.weak_strength if index == 0 else 1.0
                        keys.append(memory.write(state, row, value, strength))
                    memory.reinforce(state, keys[0], sign=1.0)
                    scores[load].append(
                        float(memory.discriminate(state, torch.stack(keys), values).mean())
                    )
                    counts[load].append(
                        sum(state.active_counts) / len(state.active_counts)
                    )
        results[name] = {
            **{
                f"load_{load}": sum(values) / len(values)
                for load, values in scores.items()
            },
            "mean": sum(
                sum(values) / len(values) for values in scores.values()
            )
            / len(scores),
            **{
                f"active_{load}": sum(values) / len(values)
                for load, values in counts.items()
            },
        }

    for name in results:
        if name != "full":
            results[name]["cost_against_full"] = (
                results[name]["mean"] - results["full"]["mean"]
            )

    output = {"config": vars(args), "documents": len(documents), "ablations": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
