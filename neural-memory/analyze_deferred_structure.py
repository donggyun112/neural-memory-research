from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor


def retention(scores: Tensor, targets: Tensor, keep: int) -> Tensor:
    """Per-episode indicator that a top-k rule keeps the write target."""
    if scores.ndim != 2 or targets.ndim != 1:
        raise ValueError("scores must be rank 2 and targets rank 1")
    if not 1 <= keep <= scores.shape[1]:
        raise ValueError("keep must lie between one and the trace count")
    return (scores.topk(keep, dim=1).indices == targets[:, None]).any(dim=1).float()


def gap_buckets(gaps: Tensor, buckets: int) -> list[tuple[int, int, Tensor]]:
    """Split episodes into equal-count bands of write-to-event distance."""
    if buckets < 1:
        raise ValueError("at least one bucket is required")
    values = gaps.float()
    edges = torch.quantile(values, torch.linspace(0.0, 1.0, buckets + 1))
    bands: list[tuple[int, int, Tensor]] = []
    for index in range(buckets):
        low, high = edges[index], edges[index + 1]
        mask = (values >= low) & (values <= high if index == buckets - 1 else values < high)
        if bool(mask.any()):
            bands.append((int(low), int(high), mask))
    return bands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training-free structure of the deferred-consolidation corpus"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--buckets", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    candidates = payload["candidates"]
    targets = payload["targets"]
    names = payload["question_type_names"]
    types = payload["question_type"]
    episodes, traces, _ = candidates.shape

    recency = torch.zeros(episodes, traces)
    recency[:, -args.keep :] = 1.0
    scores = {
        "longest_k": payload["lengths"],
        "recency_k": recency,
        "cosine_to_event": torch.einsum("etf,ef->et", candidates, payload["consolidation"]),
        "cosine_to_query": torch.einsum("etf,ef->et", candidates, payload["query"]),
    }
    baselines = {
        name: retention(value, targets, args.keep) for name, value in scores.items()
    }

    output = {
        "episodes": episodes,
        "traces": traces,
        "keep": args.keep,
        "random_capacity": args.keep / traces,
        "baselines": {
            name: {
                "all": float(value.mean()),
                **{
                    label: float(value[types == index].mean())
                    for index, label in enumerate(names)
                    if bool((types == index).any())
                },
            }
            for name, value in baselines.items()
        },
        # A finite deferral window would show the later event losing information
        # as the gap grows. Whether this corpus has one is an empirical question.
        "by_evidence_gap": [
            {
                "gap_low": low,
                "gap_high": high,
                "episodes": int(mask.sum()),
                **{name: float(value[mask].mean()) for name, value in baselines.items()},
            }
            for low, high, mask in gap_buckets(payload["evidence_gap"], args.buckets)
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
