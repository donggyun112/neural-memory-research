"""One threshold, a stream, and almost nothing worth surfacing.

Every number in this line so far came from a paired comparison: two memories,
one cue, which scores higher. That is a 50:50 problem. A memory that simply
receives data faces something else. Items arrive, one number comes out of each,
a single global threshold decides whether anything is emitted, and the fraction
of arrivals that *should* emit is small. At a base rate of one percent, a false
positive rate of one percent means one false alarm for every real one, and an
AUC of 0.99 says nothing about that.

The stream is the corpus in random order with a sliding window of recent items
as the memory. Positives are the questions, each inserted at a random point
after the turn that answers it has entered the window. Negatives are ordinary
turn arrivals, sampled to put the base rate near one percent.

Some negatives are mislabelled by construction -- a turn arriving while its own
earlier turns sit in the window arguably *should* surface something -- so the
precision reported here is a lower bound, which is the safe direction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_familiarity_at_scale import code_book


def precision_recall(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Best single global threshold, plus what full recall costs."""
    order = np.argsort(-scores)
    ranked = labels[order]
    hits = np.cumsum(ranked)
    precision = hits / np.arange(1, len(ranked) + 1)
    recall = hits / max(labels.sum(), 1)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best = int(np.argmax(f1))
    full = int(np.argmax(recall >= 1.0)) if bool((recall >= 1.0).any()) else len(ranked) - 1
    return {
        "best_f1": float(f1[best]),
        "precision_at_best_f1": float(precision[best]),
        "recall_at_best_f1": float(recall[best]),
        "precision_at_full_recall": float(precision[full]),
        "alarms_per_true_positive_at_full_recall": float(
            (full + 1 - hits[full]) / max(hits[full], 1)
        ),
        "auc": float(
            (scores[labels.astype(bool)][:, None] > scores[~labels.astype(bool)][None, :]).mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_turns.pt")
    )
    parser.add_argument("--cells", type=int, default=8192)
    parser.add_argument("--active", type=int, default=256)
    parser.add_argument("--window", type=int, default=8000)
    parser.add_argument("--base-rate", type=float, default=0.01)
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

        # Positives and negatives are the same texts. Each question fires once at
        # a point where the turn answering it sits in the window, and many times
        # at points where it does not. An earlier version used ordinary turns as
        # the negatives and every score inverted: a conversational turn arriving
        # beside its own neighbours is far more familiar than a short standalone
        # question, so the comparison measured text genre rather than memory.
        events: dict[int, list[tuple[int, float]]] = {}
        positives = 0
        for index in range(len(queries)):
            arrival = position[int(offsets[index] + targets[index])]
            low, high = arrival + 1, min(arrival + args.window, len(order) - 1)
            if low > high or low < args.window:
                continue
            events.setdefault(int(generator.integers(low, high + 1)), []).append((index, 1.0))
            positives += 1
        wanted = int(positives / args.base_rate) - positives
        drawn = 0
        while drawn < wanted:
            index = int(generator.integers(0, len(queries)))
            step = int(generator.integers(args.window, len(order)))
            arrival = position[int(offsets[index] + targets[index])]
            if step - args.window <= arrival < step:
                continue
            events.setdefault(step, []).append((index, 0.0))
            drawn += 1

        mark = np.zeros(args.cells)
        scores, labels, cosine_scores = [], [], []
        for step in range(len(order)):
            if step >= args.window:
                leaving = order[step - args.window]
                mark[indices[leaving]] -= values[leaving]
            for index, label in events.get(step, ()):
                cue_mark = np.zeros(args.cells)
                cue_mark[cue_indices[index]] = cue_values[index]
                window = order[max(0, step - args.window) : step]
                scores.append(float((cue_mark * mark).sum() / cue_mark.sum()))
                cosine_scores.append(float((turns[window] @ queries[index]).max()))
                labels.append(label)
            arriving = order[step]
            mark[indices[arriving]] += values[arriving]

        scores, labels = np.array(scores), np.array(labels)
        per_seed.append(
            {
                "seed": seed,
                "events": len(labels),
                "positives": int(labels.sum()),
                "base_rate": float(labels.mean()),
                "filter": precision_recall(scores, labels),
                "max_cosine": precision_recall(np.array(cosine_scores), labels),
            }
        )

    print(
        f"window {args.window}, {per_seed[0]['events']} arrivals, "
        f"{per_seed[0]['positives']} of them worth surfacing "
        f"(base rate {per_seed[0]['base_rate']:.4f}), {len(per_seed)} seeds\n"
    )
    print(
        f"{'reader':>12} {'AUC':>8} {'best F1':>9} {'precision':>11} {'recall':>8}"
        f" {'prec@100% recall':>17} {'false alarms each':>18}"
    )
    summary = {}
    for name in ("filter", "max_cosine"):
        cells = {
            field: float(np.mean([row[name][field] for row in per_seed]))
            for field in per_seed[0][name]
        }
        summary[name] = cells
        print(
            f"{name:>12} {cells['auc']:8.4f} {cells['best_f1']:9.4f}"
            f" {cells['precision_at_best_f1']:11.4f} {cells['recall_at_best_f1']:8.4f}"
            f" {cells['precision_at_full_recall']:17.4f}"
            f" {cells['alarms_per_true_positive_at_full_recall']:18.1f}"
        )

    output = {"config": vars(args), "per_seed": per_seed, "summary": summary}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
