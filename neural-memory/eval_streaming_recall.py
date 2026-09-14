"""A memory that is not asked anything, and has to decide when to speak.

Every measurement in this line so far compared two memories on the same cue and
asked which scored higher. A system that simply receives data and lets old
material surface has no such comparison available. One item arrives, one number
comes out, and a single global threshold decides whether anything is emitted.
That is a strictly harder problem than ranking or than a paired win rate: an AUC
of 0.99 says nothing about whether one threshold works across every context.

The stream here is the corpus in random order. Each question is presented twice
against a window of the same size: once before the turn that answers it has gone
past, and once after. Identical text, identical window length, and the only
difference is whether the memory has absorbed the thing worth surfacing. A late
arrival should fire and an early one should not.

Reported as precision and recall at one threshold chosen across all questions
together, not per question, because that is the choice the system actually has
to make.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_familiarity_at_scale import accumulate, code_book


def sweep_threshold(fires: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Best single global cut, and what it costs."""
    best = {"threshold": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    for threshold in np.unique(fires):
        predicted = fires >= threshold
        hits = float((predicted & labels).sum())
        if hits == 0:
            continue
        precision = hits / float(predicted.sum())
        recall = hits / float(labels.sum())
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_turns.pt")
    )
    parser.add_argument("--cells", type=int, default=8192)
    parser.add_argument("--active", type=int, default=256)
    parser.add_argument("--window", type=int, default=8000, help="items held in the memory")
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    queries = torch.nn.functional.normalize(payload["query"], dim=-1).numpy()
    turns = torch.nn.functional.normalize(payload["turns"], dim=-1).numpy()
    offsets = payload["offsets"].numpy()
    targets = payload["targets"].numpy()

    per_seed = []
    for seed in (int(value) for value in args.seeds.split(",")):
        generator = np.random.default_rng(seed)
        projection = generator.standard_normal(
            (turns.shape[-1], args.cells)
        ) / np.sqrt(turns.shape[-1])
        indices, values = code_book(turns, projection, args.active)
        cue_indices, cue_values = code_book(queries, projection, args.active)
        order = generator.permutation(len(turns))
        position = np.empty(len(turns), dtype=int)
        position[order] = np.arange(len(turns))

        fires, labels, cosine_fires = [], [], []
        for index in range(len(queries)):
            answering = int(offsets[index] + targets[index])
            arrival = position[answering]
            if arrival < args.window or arrival + args.window >= len(order):
                continue
            # Same window length either side of the answering turn's arrival, so
            # only its presence differs and the memory is never larger for one.
            before = order[arrival - args.window : arrival]
            after = order[arrival - args.window + 1 : arrival + 1]
            cue_mark = np.zeros(args.cells)
            cue_mark[cue_indices[index]] = cue_values[index]
            for window, label in ((after, 1.0), (before, 0.0)):
                mark = accumulate(indices, values, window, args.cells)
                fires.append(float((cue_mark * mark).sum() / cue_mark.sum()))
                cosine_fires.append(float((turns[window] @ queries[index]).max()))
                labels.append(label)

        fires = np.array(fires)
        labels = np.array(labels, dtype=bool)
        cosine_fires = np.array(cosine_fires)
        per_seed.append(
            {
                "seed": seed,
                "questions": int(labels.sum()),
                "filter": sweep_threshold(fires, labels),
                "max_cosine": sweep_threshold(cosine_fires, labels),
            }
        )

    print(
        f"window {args.window} items, {per_seed[0]['questions']} questions presented twice,"
        f" {len(per_seed)} seeds"
    )
    print("One global threshold for every question. Chance precision is 0.5.\n")
    print(f"{'reader':>12} {'threshold':>10} {'precision':>11} {'recall':>9} {'F1':>9}")
    summary = {}
    for name in ("filter", "max_cosine"):
        cells = {
            field: float(np.mean([row[name][field] for row in per_seed]))
            for field in ("threshold", "precision", "recall", "f1")
        }
        summary[name] = cells
        print(
            f"{name:>12} {cells['threshold']:10.4f} {cells['precision']:11.4f}"
            f" {cells['recall']:9.4f} {cells['f1']:9.4f}"
        )

    output = {"config": vars(args), "per_seed": per_seed, "summary": summary}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
