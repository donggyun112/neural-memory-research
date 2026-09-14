"""The same question, asked of a memory that holds the answer and one that does not.

`eval_familiarity_gate.py` compared real abstention questions against answerable
ones and could not resolve any of its differences: thirty negatives is too few,
and the two groups differ in question length as well as in content. This removes
both problems. Every answerable question becomes its own negative by writing the
same haystack with its evidence sessions taken out, and an equal number of
random distractors is dropped from the positive side so both stores hold the
same count. Identical words, identical length, identical store size; the only
difference is whether the answer is in there.

That makes the comparison paired, so the question of interest is not an AUC over
two groups but a win rate over matched pairs, with a confidence interval.

The baselines are the same ones fixed in advance, and the falsification is the
same: if the fixed-size filter does not beat the O(N) accumulating baseline, it
buys a smaller state at the cost of accuracy and adds no capability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neural_memory.familiarity import FamiliarityFilter, sparse_code


def paired_interval(
    wins: np.ndarray, seed: int, draws: int = 5000
) -> tuple[float, float, float]:
    """Win rate over matched pairs, with a resampled interval."""
    generator = np.random.default_rng(seed)
    resampled = [
        wins[generator.integers(0, len(wins), len(wins))].mean() for _ in range(draws)
    ]
    return (
        float(wins.mean()),
        float(np.percentile(resampled, 2.5)),
        float(np.percentile(resampled, 97.5)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_familiarity.pt")
    )
    parser.add_argument("--cells", type=int, default=8192)
    parser.add_argument("--active", type=int, default=128)
    parser.add_argument("--topk", default="1,2,4,8,16,32")
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    queries = torch.nn.functional.normalize(payload["query"], dim=-1).numpy()
    sessions = torch.nn.functional.normalize(payload["sessions"], dim=-1).numpy()
    offsets = payload["offsets"].numpy()
    evidence_flat = payload["evidence"].numpy()
    evidence_offsets = payload["evidence_offsets"].numpy()
    answerable = payload["answerable"].numpy()
    widths = [int(value) for value in args.topk.split(",")]

    episodes = [
        index
        for index in range(len(queries))
        if answerable[index] and evidence_offsets[index + 1] > evidence_offsets[index]
    ]
    if not episodes:
        raise RuntimeError("no answerable question carries a locatable evidence session")

    names = [
        "max_cosine",
        *[f"topk_cosine_sum_{width}" for width in widths],
        "cell_overlap",
        "cell_overlap_graded",
    ]
    wins = {name: np.zeros((len(args.seeds.split(",")), len(episodes))) for name in names}
    for row, seed in enumerate(int(value) for value in args.seeds.split(",")):
        generator = np.random.default_rng(seed)
        projection = generator.standard_normal(
            (queries.shape[-1], args.cells)
        ) / np.sqrt(queries.shape[-1])
        for column, index in enumerate(episodes):
            block = sessions[offsets[index] : offsets[index + 1]]
            marked = evidence_flat[evidence_offsets[index] : evidence_offsets[index + 1]]
            others = np.setdiff1d(np.arange(len(block)), marked)
            # Drop as many distractors from the positive side as the negative
            # loses to evidence removal, so both stores hold the same count.
            dropped = generator.choice(others, size=len(marked), replace=False)
            with_evidence = np.setdiff1d(np.arange(len(block)), dropped)
            without_evidence = others
            cue = queries[index]
            for label, kept in (("positive", with_evidence), ("negative", without_evidence)):
                held = block[kept]
                similarity = np.sort(held @ cue)[::-1]
                codes = sparse_code(held, projection, args.active)
                cue_code = sparse_code(cue[None], projection, args.active)[0]
                memory = FamiliarityFilter(args.cells)
                graded = FamiliarityFilter(args.cells, graded=True)
                for code in codes:
                    memory.write(code)
                    graded.write(code)
                scored = {
                    "max_cosine": float(similarity[0]),
                    "cell_overlap": memory.score(cue_code),
                    "cell_overlap_graded": graded.score(cue_code),
                    **{
                        f"topk_cosine_sum_{width}": float(similarity[:width].sum())
                        for width in widths
                    },
                }
                if label == "positive":
                    positive_scores = scored
                else:
                    for name in names:
                        wins[name][row, column] = float(
                            positive_scores[name] > scored[name]
                        ) + 0.5 * float(positive_scores[name] == scored[name])

    results = {}
    for name in names:
        rate, low, high = paired_interval(wins[name].mean(axis=0), 7)
        results[name] = {"win_rate": rate, "low": low, "high": high}

    best_topk = max(
        (name for name in names if name.startswith("topk_cosine_sum")),
        key=lambda name: results[name]["win_rate"],
    )
    best_filter = max(
        ("cell_overlap", "cell_overlap_graded"), key=lambda name: results[name]["win_rate"]
    )
    verdict = results[best_filter]["win_rate"] > results[best_topk]["win_rate"]

    print(
        f"{len(episodes)} matched pairs, {args.cells} cells, {args.active} active,"
        f" {len(args.seeds.split(','))} seeds"
    )
    print("Win rate: does the store that holds the answer score higher? 0.5 is nothing.\n")
    print(f"{'reader':>22} {'memory':>8} {'win rate':>10} {'95% interval':>22}")
    for name, cell in sorted(results.items(), key=lambda kv: -kv[1]["win_rate"]):
        memory = "O(1)" if name.startswith("cell_overlap") else "O(N)"
        print(
            f"{name:>22} {memory:>8} {cell['win_rate']:10.4f}"
            f"  [{cell['low']:.4f}, {cell['high']:.4f}]"
        )
    print(
        f"\nbest accumulating baseline: {best_topk} at {results[best_topk]['win_rate']:.4f}"
        f"\nbest fixed-size filter:     {best_filter} at {results[best_filter]['win_rate']:.4f}"
        f"\nfilter beats it: {verdict}"
    )

    # Comparing two confidence intervals is not a test. These readers were run on
    # the same pairs, so the difference is paired too and can be resolved far
    # more tightly than the overlap of two intervals suggests.
    head_to_head = {}
    print(f"\n{'comparison':>46} {'difference':>11} {'95% interval':>22}")
    for left, right in (
        (best_filter, best_topk),
        (best_filter, "max_cosine"),
        (best_filter, "cell_overlap"),
    ):
        if left == right:
            continue
        gap = wins[left].mean(axis=0) - wins[right].mean(axis=0)
        generator = np.random.default_rng(7)
        resampled = [
            gap[generator.integers(0, len(gap), len(gap))].mean() for _ in range(5000)
        ]
        low, high = float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))
        resolved = "resolved" if low > 0 or high < 0 else "NOT resolved"
        head_to_head[f"{left} minus {right}"] = {
            "difference": float(gap.mean()),
            "low": low,
            "high": high,
        }
        print(
            f"{left + ' minus ' + right:>46} {gap.mean():+11.4f}"
            f"  [{low:+.4f}, {high:+.4f}] {resolved}"
        )

    output = {
        "config": vars(args),
        "pairs": len(episodes),
        "results": results,
        "best_topk_baseline": best_topk,
        "best_filter": best_filter,
        "filter_beats_accumulating_baseline": verdict,
        "head_to_head": head_to_head,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
