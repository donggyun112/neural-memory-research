from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F


def combine_payloads(
    left: dict[str, Tensor | str | int],
    right: dict[str, Tensor | str | int],
) -> dict[str, Tensor | str | list[str]]:
    """Concatenate aligned frozen representations without mixing labels."""
    alignment = ("all_targets", "all_masks", "all_project_ids")
    if any(
        not torch.equal(left[name], right[name])  # type: ignore[arg-type]
        for name in alignment
    ):
        raise ValueError("feature artifacts do not describe the same episodes")
    candidates = F.normalize(
        torch.cat((left["all_candidates"], right["all_candidates"]), dim=-1),  # type: ignore[arg-type]
        dim=-1,
    )
    queries = F.normalize(
        torch.cat((left["all_queries"], right["all_queries"]), dim=-1),  # type: ignore[arg-type]
        dim=-1,
    )
    return {
        "all_candidates": candidates,
        "all_queries": queries,
        "all_targets": left["all_targets"],
        "all_masks": left["all_masks"],
        "all_project_ids": left["all_project_ids"],
        "encoder": "concat",
        "encoder_model": [str(left.get("encoder_model")), str(right.get("encoder_model"))],
        "pooling": "normalized_concat",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine aligned ContextBench features")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    left = torch.load(args.left, map_location="cpu", weights_only=True)
    right = torch.load(args.right, map_location="cpu", weights_only=True)
    payload = combine_payloads(left, right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"saved candidates={tuple(payload['all_candidates'].shape)},"
        f"queries={tuple(payload['all_queries'].shape)} to {args.output}"
    )


if __name__ == "__main__":
    main()
