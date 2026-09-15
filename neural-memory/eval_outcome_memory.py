"""A memory that remembers how things went, not only what happened.

Phase 65 found the one link between memory and a real outcome, and it was not
the one sixty-four phases were optimising: novelty predicts the next action
failing, while which item a reader surfaces says nothing. That points at a
different object. Instead of "which stored item is most relevant", ask "have I
done something like this, and did it go badly" — retrieval weighted by outcome
rather than by relevance alone.

Everything a reader sees is strictly earlier than the action being predicted, so
no outcome leaks backwards.

Baselines fixed before measuring:

  base_rate         predict the global failure rate. AUC 0.5 by construction.
  novelty           Phase 65's signal, the absence of any match in memory.
  recent_failures   the share of the last k actions that failed, ignoring
                    similarity entirely. **This is the one that matters**: if
                    failures simply cluster in time, similarity adds nothing and
                    the idea is dead.
  outcome_retrieval the proposal: similarity-weighted average of past outcomes.

Falsification: if outcome retrieval does not beat recent failures, remembering
*what* went wrong buys nothing over remembering *that* things are going wrong.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def auc(values: np.ndarray, labels: np.ndarray, draws: int = 2000) -> tuple[float, float, float]:
    positive, negative = np.flatnonzero(labels), np.flatnonzero(~labels)
    if not len(positive) or not len(negative):
        return float("nan"), float("nan"), float("nan")

    def score(left: np.ndarray, right: np.ndarray) -> float:
        difference = values[left][:, None] - values[right][None, :]
        return float((difference > 0).mean() + 0.5 * (difference == 0).mean())

    generator = np.random.default_rng(7)
    # Bootstrapping a full AUC is quadratic, so the resample is capped; the point
    # estimate still uses every position.
    cap = min(len(positive), 400), min(len(negative), 400)
    resampled = [
        score(
            generator.choice(positive, cap[0], replace=True),
            generator.choice(negative, cap[1], replace=True),
        )
        for _ in range(draws)
    ]
    return score(positive, negative), float(np.percentile(resampled, 2.5)), float(
        np.percentile(resampled, 97.5)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/claude_tool_stream.pt")
    )
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--memory-window", type=int, default=256)
    parser.add_argument("--recent", type=int, default=16, help="window for the clustering baseline")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    turns = F.normalize(payload["turns"], dim=-1)
    offsets = payload["offsets"]
    failed = payload["failed"].float()

    names = ["novelty", "recent_failures", "outcome_retrieval", "outcome_plus_novelty"]
    signals: dict[str, list[float]] = {name: [] for name in names}
    labels: list[float] = []
    for index in range(len(offsets) - 1):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        for position in range(start + args.warmup, stop - 1):
            low = max(start, position - args.memory_window)
            memory, outcomes = turns[low:position], failed[low:position]
            if not len(memory):
                continue
            scores = memory @ turns[position]
            weights = F.softmax(scores / args.temperature, dim=0)
            retrieval = float(weights @ outcomes)
            novelty = -float(scores.mean())
            labels.append(float(failed[position + 1]))
            signals["novelty"].append(novelty)
            signals["recent_failures"].append(float(outcomes[-args.recent :].mean()))
            signals["outcome_retrieval"].append(retrieval)
            # Phase 65's signal and this one answer different questions, so a
            # sum of the two is worth reporting separately.
            signals["outcome_plus_novelty"].append(retrieval + novelty)

    labels = np.array(labels, dtype=bool)
    print(
        f"{len(labels)} positions, {int(labels.sum())} followed by a failure "
        f"(base rate {labels.mean():.4f}), memory window {args.memory_window}\n"
    )
    print(f"{'signal':>22} {'AUC':>8} {'95% interval':>22}")
    results = {"positions": int(len(labels)), "failures": int(labels.sum())}
    for name in names:
        point, low, high = auc(np.array(signals[name]), labels)
        results[name] = {"auc": point, "low": low, "high": high}
        print(
            f"{name:>22} {point:8.4f}  [{low:.4f}, {high:.4f}]"
            f" {'resolved' if low > 0.5 or high < 0.5 else 'no'}"
        )
    # Comparing two intervals is not a test. Both signals are scored on the same
    # positions, so the difference between their AUCs is paired and has to be
    # resampled on shared draws.
    positive, negative = np.flatnonzero(labels), np.flatnonzero(~labels)
    generator = np.random.default_rng(7)
    cap = min(len(positive), 400), min(len(negative), 400)
    print(f"\n{'comparison':>44} {'difference':>11} {'95% interval':>22}")
    head_to_head = {}
    for left, right in (
        ("outcome_retrieval", "recent_failures"),
        ("outcome_retrieval", "novelty"),
        ("outcome_plus_novelty", "novelty"),
    ):
        first, second = np.array(signals[left]), np.array(signals[right])
        gaps = []
        for _ in range(2000):
            taken_positive = generator.choice(positive, cap[0], replace=True)
            taken_negative = generator.choice(negative, cap[1], replace=True)

            def measure(values: np.ndarray) -> float:
                difference = values[taken_positive][:, None] - values[taken_negative][None, :]
                return float((difference > 0).mean() + 0.5 * (difference == 0).mean())

            gaps.append(measure(first) - measure(second))
        gap = results[left]["auc"] - results[right]["auc"]
        low, high = float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))
        head_to_head[f"{left} minus {right}"] = {"difference": gap, "low": low, "high": high}
        print(
            f"{left + ' minus ' + right:>44} {gap:+11.4f}  [{low:+.4f}, {high:+.4f}]"
            f" {'resolved' if low > 0 or high < 0 else 'NOT resolved'}"
        )
    results["head_to_head"] = head_to_head

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
