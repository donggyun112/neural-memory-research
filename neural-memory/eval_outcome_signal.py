"""Does what the memory surfaces say anything about what actually happens?

Every number in this project is a proxy: rank a future centroid among foils.
Phase 33 retired the one endpoint that tried to connect a memory to something a
user would notice, and nothing has replaced it, so sixty phases of tuning have
been polishing a quantity with no established destination.

The action streams carry a real outcome. A tool call fails or it does not, and
the flag is recorded. This asks whether the memory a reader surfaces at position
t carries information about whether the action at t+1 fails — a question with a
label that nobody wrote for this purpose and that no amount of proxy tuning can
fake.

Readers are the same three compared everywhere else, and the reference is the
base rate: predicting failure without looking at anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    difference = positive[:, None] - negative[None, :]
    return float((difference > 0).mean() + 0.5 * (difference == 0).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/claude_tool_stream.pt")
    )
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--memory-window", type=int, default=64)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    turns = F.normalize(payload["turns"], dim=-1)
    offsets = payload["offsets"]
    if "failed" not in payload:
        raise RuntimeError("this corpus carries no outcome flags")
    failed = payload["failed"].numpy()

    # Signals available at position t, before the action at t+1 is taken.
    labels, signals = [], {name: [] for name in ("similarity", "blend", "novelty")}
    for index in range(len(offsets) - 1):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        for position in range(start + args.warmup, stop - 1):
            low = max(start, position - args.memory_window)
            memory = turns[low:position]
            if not len(memory):
                continue
            scores = memory @ turns[position]
            labels.append(float(failed[position + 1]))
            # How much the memory resembles what is happening now: the top match,
            # the softmax-weighted blend's agreement, and the absence of any
            # match at all.
            signals["similarity"].append(float(scores.max()))
            weights = F.softmax(scores / 0.02, dim=0)
            signals["blend"].append(float(weights @ scores))
            signals["novelty"].append(-float(scores.mean()))

    labels = np.array(labels, dtype=bool)
    if labels.sum() < 10:
        raise RuntimeError("too few failures to measure anything")

    results = {"positions": int(len(labels)), "failures": int(labels.sum())}
    results["base_rate"] = float(labels.mean())
    print(
        f"{len(labels)} positions, {int(labels.sum())} of them followed by a failure "
        f"(base rate {labels.mean():.4f})\n"
    )
    print(f"{'signal':>14} {'AUC':>8} {'95% interval':>22}")
    for name, values in signals.items():
        values = np.array(values)
        point = auc(values[labels], values[~labels])
        generator = np.random.default_rng(7)
        draws = []
        positive, negative = np.flatnonzero(labels), np.flatnonzero(~labels)
        for _ in range(2000):
            draws.append(
                auc(
                    values[generator.choice(positive, len(positive), replace=True)],
                    values[generator.choice(negative, len(negative), replace=True)],
                )
            )
        low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        results[name] = {"auc": point, "low": low, "high": high}
        print(
            f"{name:>14} {point:8.4f}  [{low:.4f}, {high:.4f}]"
            f" {'resolved' if low > 0.5 or high < 0.5 else 'no'}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
