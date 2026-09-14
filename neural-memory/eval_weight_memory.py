"""Storing in the function rather than in a list beside it.

Every store in this project so far scored a cue against the values it had kept,
which means it carried those values to read time. `TrainableMemory._score`
computes `reads @ values.T`, so its state was O(N) all along and it lost to an
O(N) cosine anyway. That is not a network storing anything; it is a lookup
table with a matrix in front of it.

A network stores by changing how it computes. Take the minimal version: fit a
linear map to pass the stored sessions through unchanged. The least-squares
solution is the orthogonal projector onto their span,

    W = S (S^T S)^+ S^T

which has no fitted hyperparameter, and once it exists the sessions can be
thrown away -- W is 384x384 whatever N was. Reading is `W q`: the part of the
question that the stored material can account for, which is the least-squares
combination of everything stored rather than the best single match.

There is no write gate. Phase 19 measured the ceiling on deciding what to keep
at write time and found it low; Phase 17 found deferring the decision worth
0.12. Everything is written, equally, and what survives is whatever the stored
material can express.

The task has to change with the architecture. Ranking candidates needs the
candidates present, so the question here is whether the read lands nearer the
*answer* than the raw question does -- scored against every other episode's
answer, so a reader that outputs something generically answer-shaped gains
nothing.

Baselines, fixed before measuring:

  question          the frozen question embedding, no memory at all
  top1              the single best session by cosine, the obvious retrieval
  topk_weighted     similarity-weighted sum of the top k, k swept in its favour
  projection        W q, the fitted network

Falsification: if the projection does not beat topk_weighted, fitting a
function buys nothing over keeping a weighted list.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def span_projector(sessions: np.ndarray, ridge: float) -> np.ndarray:
    """Least-squares map that passes the stored sessions through unchanged."""
    gram = sessions @ sessions.T
    solved = np.linalg.solve(gram + ridge * np.eye(len(sessions)), sessions)
    return sessions.T @ solved


def rank_of_answer(predicted: np.ndarray, answers: np.ndarray, index: int) -> int:
    norm = np.linalg.norm(predicted)
    if norm == 0:
        return len(answers) - 1
    scores = answers @ (predicted / norm)
    return int(np.flatnonzero(np.argsort(-scores) == index)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--topk", default="1,2,4,8")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    normalise = lambda tensor: torch.nn.functional.normalize(tensor, dim=-1).numpy()
    candidates = normalise(payload["candidates"])
    queries = normalise(payload["query"])
    answers = normalise(payload["answer"])
    widths = [int(value) for value in args.topk.split(",")]
    episodes, traces, _ = candidates.shape

    targets = payload["targets"].long().numpy()
    readers = [
        "question",
        *[f"topk_weighted_{width}" for width in widths],
        "projection",
        # The gate. If handing the reader the annotated evidence session does not
        # beat using no memory at all, this endpoint cannot detect a good memory
        # and nothing measured on it means anything -- the lesson of Phase 33.
        "oracle_evidence",
        "oracle_question_plus_evidence",
    ]
    ranks = {name: np.zeros(episodes) for name in readers}
    for index in range(episodes):
        block, cue = candidates[index], queries[index]
        evidence = block[int(targets[index])]
        similarity = block @ cue
        order = np.argsort(-similarity)
        predictions = {"question": cue}
        for width in widths:
            taken = order[:width]
            weights = np.maximum(0.0, similarity[taken])
            total = weights.sum()
            predictions[f"topk_weighted_{width}"] = (
                (weights[:, None] * block[taken]).sum(axis=0) / total
                if total > 0
                else cue
            )
        predictions["projection"] = span_projector(block, args.ridge) @ cue
        predictions["oracle_evidence"] = evidence
        predictions["oracle_question_plus_evidence"] = cue + evidence
        for name, predicted in predictions.items():
            ranks[name][index] = rank_of_answer(predicted, answers, index)

    results = {}
    for name, values in ranks.items():
        generator = np.random.default_rng(7)
        resampled = [
            (values[generator.integers(0, episodes, episodes)] == 0).mean()
            for _ in range(3000)
        ]
        results[name] = {
            "top1": float((values == 0).mean()),
            "top5": float((values < 5).mean()),
            "median_rank": float(np.median(values)),
            "low": float(np.percentile(resampled, 2.5)),
            "high": float(np.percentile(resampled, 97.5)),
        }

    best_topk = max(
        (name for name in results if name.startswith("topk_weighted")),
        key=lambda name: results[name]["top1"],
    )
    verdict = results["projection"]["top1"] > results[best_topk]["top1"]
    usable = (
        max(
            results["oracle_evidence"]["top1"],
            results["oracle_question_plus_evidence"]["top1"],
        )
        > results["question"]["top1"]
    )

    print(f"{episodes} episodes, {traces} sessions each, answers ranked against all {episodes}")
    print("Does the read land on this episode's own answer? Chance top-1 is "
          f"{1 / episodes:.4f}.\n")
    print(f"{'reader':>20} {'state at read':>14} {'top-1':>18} {'top-5':>8} {'median rank':>12}")
    for name, cell in sorted(results.items(), key=lambda kv: -kv[1]["top1"]):
        state = "none" if name == "question" else ("O(1)" if name == "projection" else "O(N)")
        print(
            f"{name:>20} {state:>14} {cell['top1']:10.4f}"
            f" [{cell['low']:.3f},{cell['high']:.3f}]"
            f" {cell['top5']:8.4f} {cell['median_rank']:12.1f}"
        )

    print(f"\n{'comparison':>38} {'difference':>11} {'95% interval':>22}")
    head_to_head = {}
    for left, right in (
        ("projection", best_topk),
        ("projection", "question"),
        (best_topk, "question"),
    ):
        gap = (ranks[left] == 0).astype(float) - (ranks[right] == 0).astype(float)
        generator = np.random.default_rng(7)
        resampled = [
            gap[generator.integers(0, episodes, episodes)].mean() for _ in range(3000)
        ]
        low, high = float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))
        label = f"{left} minus {right}"
        head_to_head[label] = {"difference": float(gap.mean()), "low": low, "high": high}
        resolved = "resolved" if low > 0 or high < 0 else "NOT resolved"
        print(f"{label:>38} {gap.mean():+11.4f}  [{low:+.4f}, {high:+.4f}] {resolved}")
    print(f"\nbest list baseline: {best_topk}\nprojection beats it: {verdict}")
    print(f"endpoint usable (an oracle memory beats no memory): {usable}")
    if not usable:
        print(
            "  -> the annotated evidence session does not improve on the bare question,\n"
            "     so this endpoint cannot detect a good memory and no row above is readable."
        )

    output = {
        "config": vars(args),
        "episodes": episodes,
        "endpoint_usable": usable,
        "results": results,
        "best_list_baseline": best_topk,
        "projection_beats_list": verdict,
        "head_to_head": head_to_head,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
