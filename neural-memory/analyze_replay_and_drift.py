from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from analyze_opponent_modulation import recall, write_all
from analyze_sparse_capacity import sparse_code


def replay(matrix: Tensor, tags: Tensor, selected: Tensor, rounds: int, rate: float) -> Tensor:
    """Re-apply the eligibility of chosen traces with no new input.

    Offline consolidation adds nothing the memory has not already seen; it only
    rehearses a subset. Traces left out are rehearsed zero times, so any gap
    between them is consolidation rather than fresh learning.
    """
    if rounds < 0:
        raise ValueError("rounds cannot be negative")
    for _ in range(rounds):
        matrix = matrix + rate * torch.einsum("t,tij->ij", selected, tags)
    return matrix


def reconsolidate(
    matrix: Tensor, key: Tensor, strength: float
) -> tuple[Tensor, Tensor]:
    """Read a trace and write back what was read.

    Retrieval returns a trace to a labile state, and what restabilises is what
    came back rather than what was originally stored, so whatever the read picked
    up from other traces is deposited as if it belonged to this one.

    The deposit is deliberately not error-correcting. A delta-rule update towards
    the retrieved value is exactly zero, since the state already returns it, so
    an error-correcting restabilisation could never drift at all.

    Returns the updated state and what was retrieved.
    """
    retrieved = matrix @ key
    return matrix + strength * torch.outer(F.normalize(retrieved, dim=0), key), retrieved


def drift_cycle(
    matrix: Tensor,
    keys: Tensor,
    values: Tensor,
    target: int,
    cycles: int,
    strength: float,
) -> tuple[list[float], list[float]]:
    """Repeatedly recall one trace and watch both it and its neighbours.

    A re-deposit along the retrieved direction leaves that direction unchanged,
    because the key is a unit vector and the state already answers along it. What
    moves is every other trace, through the overlap between their keys and this
    one. So repeated recall is measured on both: the recalled trace and the rest.
    """
    reference = F.normalize(values, dim=1)
    others = [index for index in range(len(keys)) if index != target]
    self_scores: list[float] = []
    other_scores: list[float] = []
    for _ in range(cycles):
        matrix, _ = reconsolidate(matrix, keys[target], strength)
        agreement = (F.normalize(keys @ matrix.T, dim=1) * reference).sum(dim=1)
        self_scores.append(float(agreement[target]))
        other_scores.append(float(agreement[others].mean()))
    return self_scores, other_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline replay, and whether reading a memory changes it"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--key-dim", type=int, default=512)
    parser.add_argument("--active", type=int, default=32)
    parser.add_argument("--load", type=int, default=24)
    parser.add_argument("--replayed", type=int, default=8)
    parser.add_argument("--replay-rounds", default="0,1,2,4")
    parser.add_argument("--replay-rate", type=float, default=0.15)
    parser.add_argument("--later-writes", type=int, default=16)
    parser.add_argument("--recall-cycles", type=int, default=6)
    parser.add_argument("--recall-strength", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    documents = F.normalize(payload["candidates"].flatten(0, 1), dim=-1)
    feature_dim = documents.shape[-1]
    rounds = [int(value) for value in args.replay_rounds.split(",")]
    width = args.load + args.later_writes

    kept: dict[str, list[float]] = {str(r): [] for r in rounds}
    skipped: dict[str, list[float]] = {str(r): [] for r in rounds}
    drifts: list[list[float]] = []
    neighbour_drifts: list[list[float]] = []

    for seed in (int(value) for value in args.seeds.split(",")):
        generator = torch.Generator().manual_seed(seed)
        key_projection = torch.randn(
            feature_dim, args.key_dim, generator=generator
        ) / args.key_dim**0.5
        value_projection = torch.randn(
            feature_dim, args.value_dim, generator=generator
        ) / args.value_dim**0.5
        for _ in range(args.episodes):
            chosen = torch.randperm(len(documents), generator=generator)[:width]
            block = documents[chosen]
            keys = sparse_code(torch.tanh(block @ key_projection), args.active)
            values = F.normalize(torch.tanh(block @ value_projection), dim=1)
            stored_keys, stored_values = keys[: args.load], values[: args.load]
            matrix, tags = write_all(stored_keys, stored_values)

            selection = torch.zeros(args.load)
            selection[
                torch.randperm(args.load, generator=generator)[: args.replayed]
            ] = 1.0
            for count in rounds:
                state = replay(matrix, tags, selection, count, args.replay_rate)
                for key, value in zip(
                    keys[args.load :], values[args.load :], strict=True
                ):
                    state = state + torch.outer(value - state @ key, key)
                scores = recall(state, stored_keys, stored_values)
                kept[str(count)].append(float(scores[selection > 0].mean()))
                skipped[str(count)].append(float(scores[selection == 0].mean()))

            target = int(torch.randint(args.load, (1,), generator=generator))
            mine, theirs = drift_cycle(
                matrix,
                stored_keys,
                stored_values,
                target,
                args.recall_cycles,
                args.recall_strength,
            )
            drifts.append(mine)
            neighbour_drifts.append(theirs)

    columns = list(zip(*drifts, strict=True))
    neighbour_columns = list(zip(*neighbour_drifts, strict=True))
    output = {
        "config": vars(args),
        "documents": len(documents),
        "replay": {
            "rehearsed": {r: sum(v) / len(v) for r, v in kept.items()},
            "skipped": {r: sum(v) / len(v) for r, v in skipped.items()},
            "gap": {
                r: sum(kept[r]) / len(kept[r]) - sum(skipped[r]) / len(skipped[r])
                for r in kept
            },
        },
        "recalled_trace_after_each_recall": {
            str(index + 1): sum(column) / len(column)
            for index, column in enumerate(columns)
        },
        "other_traces_after_each_recall": {
            str(index + 1): sum(column) / len(column)
            for index, column in enumerate(neighbour_columns)
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
