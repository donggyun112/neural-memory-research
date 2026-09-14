"""Does the fixed-size filter lose ground as the load grows, or hold it?

Phase 38 measured a graded cell accumulator against list readers at one load
and found it 0.0093 behind the best of them. That is the level. The question
this asks is the slope: the filter's state does not grow with the number of
items and the list readers' does, so what matters is whether its accuracy falls
away as N rises or stays put.

The same paired design as Phase 38 -- one memory holding the item that answers
the question, one with it removed, everything else identical -- run at loads
from a handful of items to the full turn-granularity haystack of about 491.
Phase 31's equivalent slope ran the wrong way: the store's gap to cosine grew
from 0.16 at eight documents to 0.35 at a hundred and twenty-eight. If this one
does the same the mechanism is finished; if the gap holds or narrows, the
fixed-size state is worth something that a single load could not show.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neural_memory.familiarity import FamiliarityFilter, sparse_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_turns.pt")
    )
    parser.add_argument("--cells", type=int, default=8192)
    parser.add_argument("--active", type=int, default=256)
    parser.add_argument("--loads", default="16,64,128,256,491")
    parser.add_argument("--topk", default="1,4,16,64")
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    queries = torch.nn.functional.normalize(payload["query"], dim=-1).numpy()
    turns = torch.nn.functional.normalize(payload["turns"], dim=-1).numpy()
    offsets = payload["offsets"].numpy()
    targets = payload["targets"].numpy()
    widths = [int(value) for value in args.topk.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]

    names = ["max_cosine", *[f"topk_cosine_sum_{width}" for width in widths], "cell_overlap_graded"]
    rows = []
    for load in (int(value) for value in args.loads.split(",")):
        wins = {name: [] for name in names}
        for seed in seeds:
            generator = np.random.default_rng(seed)
            projection = generator.standard_normal(
                (queries.shape[-1], args.cells)
            ) / np.sqrt(queries.shape[-1])
            for index in range(len(queries)):
                block = turns[offsets[index] : offsets[index + 1]]
                target = int(targets[index])
                if len(block) <= load:
                    continue
                others = np.setdiff1d(np.arange(len(block)), [target])
                # Both memories hold exactly `load` items. The positive holds the
                # answering turn and load-1 distractors; the negative swaps it for
                # one more distractor, so only its presence differs.
                taken = generator.choice(others, size=load, replace=False)
                positive = np.concatenate([[target], taken[: load - 1]])
                negative = taken
                cue = queries[index]
                scored = {}
                for label, kept in (("positive", positive), ("negative", negative)):
                    held = block[kept]
                    similarity = np.sort(held @ cue)[::-1]
                    memory = FamiliarityFilter(args.cells, graded=True)
                    for code in sparse_code(held, projection, args.active):
                        memory.write(code)
                    scored[label] = {
                        "max_cosine": float(similarity[0]),
                        "cell_overlap_graded": memory.score(
                            sparse_code(cue[None], projection, args.active)[0]
                        ),
                        **{
                            f"topk_cosine_sum_{width}": float(similarity[:width].sum())
                            for width in widths
                        },
                    }
                for name in names:
                    wins[name].append(
                        float(scored["positive"][name] > scored["negative"][name])
                        + 0.5 * float(scored["positive"][name] == scored["negative"][name])
                    )
        row = {"load": load, "episodes": len(wins[names[0]]) // len(seeds)}
        for name in names:
            row[name] = float(np.mean(wins[name]))
        best = max(
            (name for name in names if name != "cell_overlap_graded"),
            key=lambda name: row[name],
        )
        row["best_list_reader"] = best
        row["gap_to_best_list"] = row["cell_overlap_graded"] - row[best]
        rows.append(row)

    print(f"{len(queries)} questions, turn granularity, {args.cells} cells, {args.active} active")
    print("Win rate: does the memory holding the answering turn score higher?\n")
    header = ' '.join(f"{name.replace('topk_cosine_sum_', 'top'):>14}" for name in names)
    print(f"{'load':>6} {header} {'gap':>9}")
    for row in rows:
        cells = ' '.join(f"{row[name]:14.4f}" for name in names)
        print(f"{row['load']:>6} {cells} {row['gap_to_best_list']:+9.4f}")

    # The level at one load is what Phase 38 reported. The slope is the question
    # here, and it has to be read against each baseline separately: a reader
    # whose k grows with N is not holding its budget fixed, so beating it and
    # beating a fixed-budget reader are different claims.
    print(f"\n{'filter minus':>22} {'at load ' + str(rows[0]['load']):>12}"
          f" {'at load ' + str(rows[-1]['load']):>12} {'direction':>12}")
    slopes = {}
    for name in names:
        if name == "cell_overlap_graded":
            continue
        first = rows[0]["cell_overlap_graded"] - rows[0][name]
        last = rows[-1]["cell_overlap_graded"] - rows[-1][name]
        slopes[name] = {"first": first, "last": last}
        direction = "improving" if last > first else "worsening"
        print(f"{name:>22} {first:+12.4f} {last:+12.4f} {direction:>12}")

    output = {"config": vars(args), "rows": rows, "slopes": slopes}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
