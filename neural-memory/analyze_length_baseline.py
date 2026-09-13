from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor


def length_baseline(slot_lengths: Tensor, targets: Tensor, *, keep: int) -> dict[str, float]:
    """Score trivial selection rules that need no training.

    Candidate length is the control this project originally missed: a turn or
    session that will be revisited is also longer than one that will not, so a
    learned writer must beat "keep the longest" before its retention counts.
    """
    if slot_lengths.ndim != 2 or targets.ndim != 1:
        raise ValueError("slot_lengths must be rank 2 and targets rank 1")
    if len(slot_lengths) != len(targets):
        raise ValueError("slot_lengths and targets must describe the same episodes")
    episodes, traces = slot_lengths.shape
    if not 1 <= keep < traces:
        raise ValueError("keep must be at least one and smaller than the trace count")
    gold = targets[:, None]

    def hit(indices: Tensor) -> float:
        return float((indices == gold).any(dim=1).float().mean())

    target_lengths = slot_lengths.gather(1, gold).squeeze(1)
    other_total = slot_lengths.sum(dim=1) - target_lengths
    return {
        "episodes": float(episodes),
        "traces": float(traces),
        "random_expectation": keep / traces,
        "longest_hit": hit(slot_lengths.topk(keep, dim=1).indices),
        "shortest_hit": hit((-slot_lengths).topk(keep, dim=1).indices),
        "recency_hit": hit(
            torch.arange(traces - keep, traces).expand(episodes, keep)
        ),
        "target_mean_chars": float(target_lengths.mean()),
        "other_mean_chars": float((other_total / (traces - 1)).mean()),
    }


def claude_slot_lengths(history: Path, candidates: int) -> dict[str, tuple[Tensor, Tensor]]:
    """Recompute candidate turn lengths from local history without storing text."""
    from neural_memory.claude_logs import (
        iter_history_conversations,
        iter_revisit_examples,
        stable_eval_split,
    )

    rows: dict[str, list[tuple[list[float], int]]] = {"train": [], "eval": []}
    for conversation in iter_history_conversations(history, gap_seconds=6 * 60 * 60):
        split = "eval" if stable_eval_split(conversation.project) else "train"
        for example in iter_revisit_examples(
            conversation,
            candidates=candidates,
            min_delay=2,
            min_overlap=2.0,
            min_margin=0.5,
        ):
            rows[split].append(
                (
                    [float(len(conversation.turns[index])) for index in example.candidate_indices],
                    example.target_offset,
                )
            )
    return {
        split: (
            torch.tensor([lengths for lengths, _ in entries]),
            torch.tensor([target for _, target in entries]),
        )
        for split, entries in rows.items()
        if entries
    }


def longmemeval_slot_lengths(path: Path, candidates: int) -> dict[str, tuple[Tensor, Tensor]]:
    from neural_memory.longmemeval import iter_longmemeval_revisits

    rows: dict[str, list[tuple[list[float], int]]] = {"train": [], "eval": []}
    for example in iter_longmemeval_revisits(path, candidates=candidates):
        rows[example.split].append(
            ([float(len(text)) for text in example.candidates], example.target_offset)
        )
    return {
        split: (
            torch.tensor([lengths for lengths, _ in entries]),
            torch.tensor([target for _, target in entries]),
        )
        for split, entries in rows.items()
        if entries
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training-free length and recency baselines for write selection"
    )
    parser.add_argument("--source", choices=("claude", "longmemeval"), default="claude")
    parser.add_argument(
        "--history", type=Path, default=Path.home() / ".claude" / "history.jsonl"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/longmemeval_s_cleaned.json")
    )
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.source == "claude":
        splits = claude_slot_lengths(args.history, args.candidates)
    else:
        splits = longmemeval_slot_lengths(args.input, args.candidates)
    output = {
        "source": args.source,
        "keep": args.keep,
        "splits": {
            split: length_baseline(lengths, targets, keep=args.keep)
            for split, (lengths, targets) in splits.items()
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
