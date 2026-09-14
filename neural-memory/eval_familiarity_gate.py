"""Can a fixed-size familiarity signal tell "I never stored that" from "I did"?

The baselines are fixed before the measurement, and two of them are the kind
that overturned Phase 34 and Phase 36:

  question_length   no memory at all. Abstention questions in this corpus are
                    shorter on average (62.5 against 86.4 characters), so any
                    reader has to beat the confound before it means anything.
  max_cosine        O(N) memory, the obvious way to gate: is anything stored
                    close to the cue?
  topk_cosine_sum   O(N) memory, the obvious way to accumulate. Its k is swept
                    and resolved in its own favour, on the same data.
  cell_overlap      O(1) memory. The proposal: mark the cells each document
                    uses and ask how much of the cue's own weight lands on
                    cells already marked.

AUC is threshold-free, so no decision boundary is tuned for anyone.

Falsification, stated in advance: if cell_overlap does not beat
topk_cosine_sum, the mechanism buys a smaller state at the cost of accuracy and
adds no capability, and this line closes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neural_memory.familiarity import FamiliarityFilter, auc, sparse_code


def bootstrap_difference(
    first: np.ndarray, second: np.ndarray, answerable: np.ndarray, seed: int, draws: int = 5000
) -> tuple[float, float, float]:
    """Confidence interval on the AUC gap between two readers.

    Thirty abstention questions is few, and a table of point estimates hides
    that. Resampling both groups says whether the ordering in the table is
    something the data can actually resolve.
    """
    generator = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(answerable), np.flatnonzero(~answerable)
    gaps = np.empty(draws)
    for draw in range(draws):
        taken_positive = generator.choice(positive, size=len(positive), replace=True)
        taken_negative = generator.choice(negative, size=len(negative), replace=True)
        gaps[draw] = auc(first[taken_positive], first[taken_negative]) - auc(
            second[taken_positive], second[taken_negative]
        )
    return float(gaps.mean()), float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def cosine_readers(sessions: np.ndarray, cue: np.ndarray, widths: list[int]) -> dict[str, float]:
    similarity = np.sort(sessions @ cue)[::-1]
    scores = {"max_cosine": float(similarity[0])}
    for width in widths:
        scores[f"topk_cosine_sum_{width}"] = float(similarity[:width].sum())
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_familiarity.pt")
    )
    # The baseline gets its best k, so the proposal gets its best filter shape.
    # Phase 37's formula says a filter holding ~48 documents saturates unless
    # k/m stays near 0.007, so the sweep has to reach settings that sparse.
    parser.add_argument("--cells", default="2048,8192")
    parser.add_argument("--actives", default="8,16,32,64,128")
    parser.add_argument("--topk", default="2,4,8,16,32")
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    queries = torch.nn.functional.normalize(payload["query"], dim=-1).numpy()
    sessions = torch.nn.functional.normalize(payload["sessions"], dim=-1).numpy()
    offsets = payload["offsets"].numpy()
    answerable = payload["answerable"].numpy()
    lengths = payload["question_chars"].numpy()
    widths = [int(value) for value in args.topk.split(",")]

    fixed = {"question_length": lengths}
    for name in ["max_cosine", *[f"topk_cosine_sum_{width}" for width in widths]]:
        fixed[name] = np.zeros(len(queries))
    for index in range(len(queries)):
        block = sessions[offsets[index] : offsets[index + 1]]
        for name, value in cosine_readers(block, queries[index], widths).items():
            fixed[name][index] = value

    seeds = [int(value) for value in args.seeds.split(",")]
    results = {
        name: {"mean": float(auc(value[answerable], value[~answerable])), "std": 0.0}
        for name, value in fixed.items()
    }
    marked_share: dict[str, float] = {}
    for cells in (int(value) for value in args.cells.split(",")):
        for active in (int(value) for value in args.actives.split(",")):
            if active >= cells:
                continue
            per_seed, saturation = [], []
            for seed in seeds:
                generator = np.random.default_rng(seed)
                projection = generator.standard_normal(
                    (queries.shape[-1], cells)
                ) / np.sqrt(queries.shape[-1])
                scores = np.zeros(len(queries))
                for index in range(len(queries)):
                    block = sessions[offsets[index] : offsets[index + 1]]
                    memory = FamiliarityFilter(cells)
                    for code in sparse_code(block, projection, active):
                        memory.write(code)
                    saturation.append(float(memory.mark.mean()))
                    scores[index] = memory.score(
                        sparse_code(queries[index][None], projection, active)[0]
                    )
                per_seed.append(auc(scores[answerable], scores[~answerable]))
            name = f"cell_overlap_{cells}x{active}"
            results[name] = {
                "mean": float(np.mean(per_seed)),
                "std": float(np.std(per_seed)),
            }
            marked_share[name] = float(np.mean(saturation))
            # Keep the last seed's per-question scores so the winning shape can
            # be compared against a baseline question by question.
            fixed[name] = scores
    best_topk = max(
        (name for name in results if name.startswith("topk_cosine_sum")),
        key=lambda name: results[name]["mean"],
    )
    best_overlap = max(
        (name for name in results if name.startswith("cell_overlap")),
        key=lambda name: results[name]["mean"],
    )
    verdict = results[best_overlap]["mean"] > results[best_topk]["mean"]

    print(
        f"{int(answerable.sum())} answerable vs {int((~answerable).sum())} abstention questions,"
        f" {len(sessions)} sessions"
    )
    print("AUC that an answerable question scores above an abstention one; 0.5 is nothing.\n")
    print(f"{'reader':>26} {'memory':>8} {'AUC':>18} {'cells marked':>13}")
    memory_of = lambda name: "none" if name == "question_length" else (
        "O(1)" if name.startswith("cell_overlap") else "O(N)"
    )
    for name, cell in sorted(results.items(), key=lambda kv: -kv[1]["mean"]):
        share = marked_share.get(name)
        tail = f"{share:13.3f}" if share is not None else " " * 13
        print(f"{name:>26} {memory_of(name):>8} {cell['mean']:12.4f}±{cell['std']:.4f}{tail}")
    print(
        f"\nbest accumulating baseline: {best_topk} at {results[best_topk]['mean']:.4f}"
        f"\nbest filter shape:          {best_overlap} at {results[best_overlap]['mean']:.4f}"
        f"\nfilter beats it: {verdict}"
    )
    comparisons = {}
    print(f"\n{'comparison':>44} {'AUC gap':>9} {'95% interval':>22}")
    for left, right in (
        (best_overlap, best_topk),
        (best_overlap, "question_length"),
        ("max_cosine", "question_length"),
    ):
        gap, low, high = bootstrap_difference(
            fixed[left], fixed[right], answerable, seeds[0]
        )
        resolved = "resolved" if low > 0 or high < 0 else "NOT resolved"
        comparisons[f"{left} minus {right}"] = {"gap": gap, "low": low, "high": high}
        print(f"{left + ' minus ' + right:>44} {gap:+9.4f}  [{low:+.4f}, {high:+.4f}] {resolved}")

    output = {
        "config": vars(args),
        "questions": len(queries),
        "abstention": int((~answerable).sum()),
        "results": results,
        "marked_share": marked_share,
        "best_topk_baseline": best_topk,
        "best_filter": best_overlap,
        "filter_beats_accumulating_baseline": verdict,
        "bootstrap": comparisons,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
