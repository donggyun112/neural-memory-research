"""The answer key a memory can get for free: what happens next.

Every evaluation in this line has needed a human annotation -- which session
holds the answer, which question is unanswerable -- and there are 52 of those.
Modern deep learning did not scale on annotations; it scaled on targets the data
manufactures for itself, one per position. A memory has the same thing available
and this project never used it: at time t, what should surface is whatever turns
out to matter at t+1, and the stream says what that was.

That gives a target at every position instead of at 52 of them, it is continuous
rather than a binary "should fire", and it needs nobody to label anything.

One correction it forces. Phase 43 shuffled the stream, which destroys the only
structure this objective runs on -- in a random order nothing predicts anything.
Turns are streamed here in their true order, sessions in haystack order and
messages within a session in sequence.

Scoring: at position t the memory holds every earlier turn and each reader
surfaces one of them. The surfaced item is then asked to pick the real future
out of a hundred candidate futures drawn from elsewhere in the corpus, so the
number means the same thing at every position.

  recency          surface the most recent turn. In conversation the near future
                   is mostly the present topic, so this is the baseline to beat.
  current_cosine   surface whatever in memory is most like the current turn --
                   association, with no notion of what is coming.
  random           chance.
  oracle           surface whatever in memory best matches the real future. Not
                   a competitor; the ceiling on what surfacing one stored item
                   can buy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_turns.pt")
    )
    parser.add_argument("--horizon", type=int, default=5, help="turns ahead that count as future")
    parser.add_argument("--warmup", type=int, default=32, help="turns before scoring starts")
    parser.add_argument("--positions", type=int, default=40, help="scored positions per question")
    parser.add_argument("--foils", type=int, default=99)
    parser.add_argument("--distant", type=int, default=64, help="how far back counts as old")
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    turns = normalise(payload["turns"].numpy())
    offsets = payload["offsets"].numpy()

    # Future centroids are precomputed for every position so foils can be drawn
    # from anywhere in the corpus without recomputing them.
    futures = np.zeros_like(turns)
    valid = np.zeros(len(turns), dtype=bool)
    for index in range(len(offsets) - 1):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        for position in range(start, stop - args.horizon):
            futures[position] = turns[position + 1 : position + 1 + args.horizon].mean(axis=0)
            valid[position] = True
    futures = normalise(futures)
    pool = np.flatnonzero(valid)

    names = ["random", "recency", "current_cosine", "oracle"]
    rows = []
    for seed in (int(value) for value in args.seeds.split(",")):
        generator = np.random.default_rng(seed)
        ranks = {name: [] for name in names}
        distant_ranks = {name: [] for name in names}
        for index in range(len(offsets) - 1):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            usable = np.arange(start + args.warmup, stop - args.horizon)
            if len(usable) < 1:
                continue
            chosen = generator.choice(
                usable, size=min(args.positions, len(usable)), replace=False
            )
            for position in chosen:
                memory = turns[start:position]
                future = futures[position]
                picks = {
                    "random": int(generator.integers(0, len(memory))),
                    "recency": len(memory) - 1,
                    "current_cosine": int(np.argmax(memory @ turns[position])),
                    "oracle": int(np.argmax(memory @ future)),
                }
                foils = generator.choice(pool, size=args.foils, replace=False)
                candidates = np.vstack([future, futures[foils]])
                # A position is "old" when even the best available item sits far
                # back, which is where a memory has to do something recency
                # cannot. No annotation is needed to find these.
                old = (len(memory) - 1 - picks["oracle"]) >= args.distant
                for name, pick in picks.items():
                    scores = candidates @ memory[pick]
                    rank = int((scores > scores[0]).sum())
                    ranks[name].append(rank)
                    if old:
                        distant_ranks[name].append(rank)
        row = {"seed": seed, "positions": len(ranks["random"]),
               "old_positions": len(distant_ranks["random"])}
        for name in names:
            values = np.array(ranks[name])
            row[f"{name}_top1"] = float((values == 0).mean())
            row[f"{name}_top10"] = float((values < 10).mean())
            old = np.array(distant_ranks[name])
            row[f"{name}_top1_old"] = float((old == 0).mean()) if len(old) else float("nan")
        rows.append(row)

    print(
        f"{rows[0]['positions']} scored positions, horizon {args.horizon}, "
        f"1-in-{args.foils + 1} choice, {len(rows)} seeds"
    )
    print(
        f"Chance top-1 is {1 / (args.foils + 1):.4f}. "
        f"{rows[0]['old_positions']} positions need something older than {args.distant} turns.\n"
    )
    # The headline comparison is on the old positions, and 521 of them across
    # three seeds is few enough that the gap needs an interval rather than a
    # point estimate.
    generator = np.random.default_rng(7)
    paired = {
        name: np.concatenate([np.array(block[name]) for block in (distant_ranks,)])
        for name in names
    }
    gap = (paired["current_cosine"] == 0).astype(float) - (paired["recency"] == 0).astype(float)
    draws = [gap[generator.integers(0, len(gap), len(gap))].mean() for _ in range(5000)]
    low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    print(f"{'reader':>16} {'top-1':>9} {'top-10':>9} {'top-1, old only':>17}")
    summary = {}
    for name in names:
        cells = {
            field: float(np.mean([row[f"{name}_{field}"] for row in rows]))
            for field in ("top1", "top10", "top1_old")
        }
        summary[name] = cells
        print(
            f"{name:>16} {cells['top1']:9.4f} {cells['top10']:9.4f} {cells['top1_old']:17.4f}"
        )

    print(
        f"\nassociation minus recency on old positions: {gap.mean():+.4f}"
        f"  [{low:+.4f}, {high:+.4f}] {'resolved' if low > 0 or high < 0 else 'NOT resolved'}"
        f"\nheadroom to the oracle there: "
        f"{summary['oracle']['top1_old'] - summary['current_cosine']['top1_old']:+.4f}"
    )

    output = {
        "config": vars(args),
        "per_seed": rows,
        "summary": summary,
        "association_minus_recency_old": {
            "difference": float(gap.mean()),
            "low": low,
            "high": high,
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
