from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from analyze_sparse_capacity import sparse_code


def write(matrix: Tensor, key: Tensor, value: Tensor, strength: float, decay: float) -> Tensor:
    """One decayed delta-rule update."""
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be inside (0, 1]")
    return decay * matrix + strength * torch.outer(value - matrix @ key, key)


def run_conflict(
    cue: Tensor,
    early: Tensor,
    late: Tensor,
    filler_keys: Tensor,
    filler_values: Tensor,
    *,
    delay: int,
    fast: tuple[float, float],
    slow: tuple[float, float],
    single: bool = False,
) -> tuple[float, float]:
    """Teach one cue two different answers, then read the sum after a delay.

    The fast store writes strongly and forgets quickly, the slow store writes
    weakly and holds, and the readout is their sum. Both are written at the same
    moment, so nothing is promoted from one into the other. If the expressed
    answer changes with delay, it is because the two stores fade at different
    rates rather than because anything moved between them.

    Returns agreement with the early answer and with the late one.
    """
    if delay > len(filler_keys):
        raise ValueError("not enough filler documents for this delay")
    fast_strength, fast_decay = fast
    slow_strength, slow_decay = slow
    quick = torch.zeros(early.shape[0], cue.shape[0])
    lasting = torch.zeros(early.shape[0], cue.shape[0])
    if single:
        # The honest single-store control holds both answers in one matrix, where
        # the later write simply supersedes the earlier one on the same key.
        quick = write(quick, cue, early, fast_strength, fast_decay)
        quick = write(quick, cue, late, fast_strength, fast_decay)
    else:
        quick = write(quick, cue, early, fast_strength, fast_decay)
        lasting = write(lasting, cue, late, slow_strength, slow_decay)
    for index in range(delay):
        key, value = filler_keys[index], filler_values[index]
        quick = write(quick, key, value, fast_strength, fast_decay)
        if not single:
            lasting = write(lasting, key, value, slow_strength, slow_decay)
    read = quick @ cue if single else quick @ cue + lasting @ cue
    normalised = F.normalize(read, dim=0)
    return (
        float(normalised @ F.normalize(early, dim=0)),
        float(normalised @ F.normalize(late, dim=0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Does the expressed answer flip as the fast store fades?"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--key-dim", type=int, default=512)
    parser.add_argument("--active", type=int, default=32)
    parser.add_argument("--fast", default="1.0,0.75", help="strength,decay of the fast store")
    parser.add_argument("--slow", default="0.25,0.99", help="strength,decay of the slow store")
    parser.add_argument("--delays", default="0,1,2,4,8,16,32")
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    documents = F.normalize(payload["candidates"].flatten(0, 1), dim=-1)
    feature_dim = documents.shape[-1]
    fast = tuple(float(value) for value in args.fast.split(","))
    slow = tuple(float(value) for value in args.slow.split(","))
    delays = [int(value) for value in args.delays.split(",")]
    width = max(delays) + 3

    curves: dict[str, dict[str, dict[str, float]]] = {}
    for label, single in (("two_stores", False), ("fast_store_only", True)):
        early: dict[str, list[float]] = {str(d): [] for d in delays}
        late: dict[str, list[float]] = {str(d): [] for d in delays}
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
                for delay in delays:
                    agreement = run_conflict(
                        keys[0],
                        values[1],
                        values[2],
                        keys[3:],
                        values[3:],
                        delay=delay,
                        fast=fast,
                        slow=slow,
                        single=single,
                    )
                    early[str(delay)].append(agreement[0])
                    late[str(delay)].append(agreement[1])
        curves[label] = {
            "early_answer": {d: sum(v) / len(v) for d, v in early.items()},
            "late_answer": {d: sum(v) / len(v) for d, v in late.items()},
            "margin": {
                d: sum(early[d]) / len(early[d]) - sum(late[d]) / len(late[d])
                for d in early
            },
        }

    output = {
        "config": vars(args),
        "documents": len(documents),
        "curves": curves,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
