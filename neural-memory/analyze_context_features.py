from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure raw ContextBench feature separation")
    parser.add_argument("features", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.features:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        candidates = payload["all_candidates"]
        queries = payload["all_queries"]
        targets = payload["all_targets"]
        masks = payload["all_masks"]
        cosine = (candidates * queries[:, None]).sum(dim=-1)
        positive = cosine[targets & masks]
        negative = cosine[~targets & masks]
        top1 = cosine.masked_fill(~masks, -torch.inf).argmax(dim=-1)
        top1_hit = targets.gather(1, top1[:, None]).float().mean()
        valid_candidates = candidates[masks]
        print(
            f"{path}:pos={positive.mean():.6f},neg={negative.mean():.6f},"
            f"margin={positive.mean() - negative.mean():.6f},top1={top1_hit:.6f},"
            f"candidate_mean_norm={valid_candidates.mean(dim=0).norm():.6f},"
            f"query_mean_norm={queries.mean(dim=0).norm():.6f}"
        )


if __name__ == "__main__":
    main()
