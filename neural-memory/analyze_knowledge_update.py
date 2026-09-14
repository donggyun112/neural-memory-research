from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch import Tensor
from torch.nn import functional as F

from analyze_sparse_capacity import sparse_code


def write_store(keys: Tensor, values: Tensor) -> Tensor:
    """Write documents in order under the delta rule and return the state.

    The rule subtracts what the state already returns for a key before adding,
    so a second document landing on the same units replaces what the first one
    put there. That is the only property being tested here.
    """
    matrix = values.new_zeros(values.shape[1], keys.shape[1])
    for key, value in zip(keys, values, strict=True):
        matrix = matrix + torch.outer(value - matrix @ key, key)
    return matrix


def read_scores(matrix: Tensor, cue: Tensor, values: Tensor) -> Tensor:
    read = F.normalize(matrix @ cue, dim=0)
    return values @ read


def key_overlap(keys: Tensor, first: int, second: int) -> float:
    """Fraction of the superseded statement's active units the later one reuses."""
    a = set(keys[first].nonzero().flatten().tolist())
    b = set(keys[second].nonzero().flatten().tolist())
    return len(a & b) / max(len(a), 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Can the store tell a current fact from the one it replaced?"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_updates.pt")
    )
    parser.add_argument("--key-dim", type=int, default=512)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--active", type=int, default=32)
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    candidates = F.normalize(payload["candidates"], dim=-1)
    queries = F.normalize(payload["query"], dim=-1)
    targets = payload["targets"].long()
    superseded = payload["superseded"].long()
    episodes, traces, features = candidates.shape

    # Order-free references. Cosine cannot express which statement came last, and
    # recency can express nothing else, so they bracket what the store must beat.
    cosine = torch.einsum('end,ed->en', candidates, queries)
    positions = torch.arange(traces).float().expand(episodes, traces) / (traces - 1)
    rank = torch.arange(episodes)

    def score_report(scores: Tensor) -> dict[str, float]:
        return {
            "hit_rate": float((scores.argmax(dim=-1) == targets).float().mean()),
            "prefers_current": float(
                (scores[rank, targets] > scores[rank, superseded]).float().mean()
            ),
        }

    # A store with no decay is also recency-biased, so beating cosine is not
    # enough: the comparison has to be against cosine that has been told about
    # order. The sweep is resolved in the baseline's favour, taking whichever
    # tilt scores best rather than a tilt fixed in advance.
    tilts = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    tilted = [(weight, score_report(cosine + weight * positions)) for weight in tilts]
    best_weight, best_tilted = max(tilted, key=lambda pair: pair[1]["hit_rate"])
    # The other obvious heuristic: shortlist by topic, then take the later one.
    shortlist = cosine.topk(2, dim=-1).indices
    later = shortlist.max(dim=-1).values
    references = {
        "chance": {
            "hit_rate": 1.0 / traces,
            "prefers_current": 0.5,
        },
        "cosine_with_recency_tilt": {
            **best_tilted,
            "weight": best_weight,
            "swept": {str(weight): report["hit_rate"] for weight, report in tilted},
        },
        "cosine_top2_take_later": {
            "hit_rate": float((later == targets).float().mean()),
            # Trivially 1.0, like recency: the rule takes the later index and the
            # target is by definition the later statement. Reported for symmetry,
            # not as evidence of anything.
            "prefers_current": 1.0,
            "both_statements_shortlisted": float(
                (
                    (shortlist == targets.unsqueeze(-1)).any(dim=-1)
                    & (shortlist == superseded.unsqueeze(-1)).any(dim=-1)
                ).float().mean()
            ),
        },
        "cosine": score_report(cosine),
        "recency": {
            "hit_rate": float((targets == traces - 1).float().mean()),
            "prefers_current": float((targets > superseded).float().mean()),
        },
    }

    rows: list[dict[str, float]] = []
    for seed in (int(value) for value in args.seeds.split(",")):
        generator = torch.Generator().manual_seed(seed)
        key_projection = torch.randn(
            features, args.key_dim, generator=generator
        ) / args.key_dim**0.5
        value_projection = torch.randn(
            features, args.value_dim, generator=generator
        ) / args.value_dim**0.5
        forward_hits, forward_current, reverse_current, overlaps = [], [], [], []
        for index in range(episodes):
            block = candidates[index]
            keys = sparse_code(torch.tanh(block @ key_projection), args.active)
            values = F.normalize(torch.tanh(block @ value_projection), dim=1)
            cue = sparse_code(
                torch.tanh(queries[index] @ key_projection).unsqueeze(0), args.active
            )[0]
            target, older = int(targets[index]), int(superseded[index])
            overlaps.append(key_overlap(keys, older, target))

            scores = read_scores(write_store(keys, values), cue, values)
            forward_hits.append(float(int(scores.argmax()) == target))
            forward_current.append(float(scores[target] > scores[older]))

            # The decisive control: writing the same documents in the opposite
            # order leaves every embedding untouched, so anything order-free
            # cannot move. If the preference comes from overwriting, it flips.
            backward = torch.arange(traces - 1, -1, -1)
            reversed_scores = read_scores(
                write_store(keys[backward], values[backward]), cue, values
            )
            reverse_current.append(float(reversed_scores[target] > reversed_scores[older]))
        rows.append(
            {
                "hit_rate": mean(forward_hits),
                "prefers_current": mean(forward_current),
                "prefers_current_reversed": mean(reverse_current),
                "key_overlap": mean(overlaps),
            }
        )

    output = {
        "config": vars(args),
        "episodes": episodes,
        "traces": traces,
        "references": references,
        "store": {
            metric: {
                "mean": mean(row[metric] for row in rows),
                "std": pstdev(row[metric] for row in rows),
            }
            for metric in rows[0]
        },
        "per_seed": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
