"""The slope, taken out to the load the argument is actually about.

Phase 41 ran from 16 to 491 items -- one question's haystack -- and found the
fixed-size filter almost flat while maximum cosine fell from 0.9006 to 0.5714.
491 is not the regime the scaling claim is about. Pooling every question's turns
gives 25,552 items to draw from, so the same paired comparison runs two orders
of magnitude further.

Each question keeps all of its own turns in both memories, so the hard
distractors -- the same conversation, the same person -- are always present and
only the easy ones scale. The positive memory additionally holds the turn that
answers the question; the negative holds one more unrelated turn in its place.

Codes are computed once for every turn and a memory is then the sum of the rows
it holds, which is what makes the large loads tractable: the graded filter is
linear in what it stores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def code_book(turns: np.ndarray, projection: np.ndarray, active: int) -> tuple[np.ndarray, np.ndarray]:
    """Sparse code every turn once, as indices and values."""
    expanded = np.maximum(0.0, turns @ projection)
    indices = np.argpartition(expanded, -active, axis=1)[:, -active:]
    values = np.take_along_axis(expanded, indices, axis=1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    return indices, values


def accumulate(indices: np.ndarray, values: np.ndarray, rows: np.ndarray, cells: int) -> np.ndarray:
    return np.bincount(
        indices[rows].ravel(), weights=values[rows].ravel(), minlength=cells
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_turns.pt")
    )
    parser.add_argument("--cells", type=int, default=8192)
    parser.add_argument("--active", type=int, default=256)
    parser.add_argument("--loads", default="491,2000,8000,25000")
    parser.add_argument("--topk", default="16,64,256")
    parser.add_argument(
        "--fill",
        choices=("random", "nearest"),
        default="random",
        help="random draws distant clutter; nearest fills with the turns most like the cue",
    )
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    queries = torch.nn.functional.normalize(payload["query"], dim=-1).numpy()
    turns = torch.nn.functional.normalize(payload["turns"], dim=-1).numpy()
    offsets = payload["offsets"].numpy()
    targets = payload["targets"].numpy()
    widths = [int(value) for value in args.topk.split(",")]
    loads = [int(value) for value in args.loads.split(",")]
    names = ["max_cosine", *[f"topk_cosine_sum_{width}" for width in widths], "filter"]

    rows_out: list[dict] = []
    per_question: dict[int, list[dict[str, np.ndarray]]] = {}
    for seed in (int(value) for value in args.seeds.split(",")):
        generator = np.random.default_rng(seed)
        projection = generator.standard_normal(
            (turns.shape[-1], args.cells)
        ) / np.sqrt(turns.shape[-1])
        indices, values = code_book(turns, projection, args.active)
        cue_indices, cue_values = code_book(queries, projection, args.active)
        for load in loads:
            if load > len(turns):
                continue
            wins = {name: [] for name in names}
            for index in range(len(queries)):
                own = np.arange(offsets[index], offsets[index + 1])
                target = int(offsets[index] + targets[index])
                if load < len(own):
                    continue
                # Every one of this question's own turns stays in both memories,
                # so the distractors that actually resemble the answer are never
                # the ones being scaled away.
                outside = np.setdiff1d(np.arange(len(turns)), own)
                # What stands in for the answering turn is drawn at random and
                # does not depend on the load or the fill mode, so the swap is
                # equally hard everywhere and only the clutter around it varies.
                # Two earlier versions got this wrong: one took the stand-in from
                # the end of the fill, which made it easier as the fill grew, and
                # one made it the cue's nearest neighbour, which made the test
                # ask whether the answer is the nearest turn rather than whether
                # the memory holds it.
                swap = int(generator.choice(outside))
                rest = outside[outside != swap]
                wanted = load - len(own)
                if args.fill == "nearest":
                    closeness = turns[rest] @ queries[index]
                    fill = rest[np.argsort(-closeness)[:wanted]]
                else:
                    fill = generator.choice(rest, size=wanted, replace=False)
                positive = np.concatenate([own, fill])
                negative = np.concatenate([own[own != target], [swap], fill])
                cue = queries[index]
                cue_mark = np.zeros(args.cells)
                cue_mark[cue_indices[index]] = cue_values[index]
                scored = {}
                for label, kept in (("positive", positive), ("negative", negative)):
                    similarity = np.sort(turns[kept] @ cue)[::-1]
                    mark = accumulate(indices, values, kept, args.cells)
                    scored[label] = {
                        "max_cosine": float(similarity[0]),
                        "filter": float((cue_mark * mark).sum() / cue_mark.sum()),
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
            rows_out.append(
                {"seed": seed, "load": load, "episodes": len(wins["filter"]),
                 **{name: float(np.mean(wins[name])) for name in names}}
            )
            per_question.setdefault(load, []).append(
                {name: np.array(wins[name]) for name in names}
            )

    table = {}
    for load in loads:
        matching = [row for row in rows_out if row["load"] == load]
        if matching:
            table[load] = {
                name: float(np.mean([row[name] for row in matching])) for name in names
            }

    print(f"{len(queries)} questions, {len(turns)} turns pooled, "
          f"{args.cells} cells, {args.active} active")
    print("Win rate: does the memory holding the answering turn score higher?\n")
    header = ' '.join(f"{name.replace('topk_cosine_sum_', 'top'):>13}" for name in names)
    print(f"{'load':>7} {header}")
    for load, cells in table.items():
        print(f"{load:>7} " + ' '.join(f"{cells[name]:13.4f}" for name in names))

    first, last = min(table), max(table)
    print(f"\n{'filter minus':>22} {'at ' + str(first):>12} {'at ' + str(last):>12} {'direction':>12}")
    slopes = {}
    for name in names:
        if name == "filter":
            continue
        low = table[first]["filter"] - table[first][name]
        high = table[last]["filter"] - table[last][name]
        slopes[name] = {"low": low, "high": high}
        print(f"{name:>22} {low:+12.4f} {high:+12.4f}"
              f" {'improving' if high > low else 'worsening':>12}")

    # Fifty-two questions is few, and a difference of 0.08 is four of them. The
    # comparison is paired -- every reader saw the same questions -- so the gap
    # is resampled question by question rather than compared across two means.
    print(f"\n{'comparison at the largest load':>38} {'difference':>11} {'95% interval':>22}")
    resolved_gaps = {}
    biggest = max(table)
    stacked = {
        name: np.mean([block[name] for block in per_question[biggest]], axis=0)
        for name in names
    }
    for name in names:
        if name == "filter":
            continue
        gap = stacked["filter"] - stacked[name]
        generator = np.random.default_rng(7)
        draws = [gap[generator.integers(0, len(gap), len(gap))].mean() for _ in range(5000)]
        low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        resolved_gaps[name] = {"difference": float(gap.mean()), "low": low, "high": high}
        print(
            f"{'filter minus ' + name:>38} {gap.mean():+11.4f}  [{low:+.4f}, {high:+.4f}]"
            f" {'resolved' if low > 0 or high < 0 else 'NOT resolved'}"
        )

    output = {
        "config": vars(args),
        "per_seed": rows_out,
        "table": table,
        "slopes": slopes,
        "largest_load_gaps": resolved_gaps,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
