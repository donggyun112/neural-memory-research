from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from analyze_sparse_capacity import sparse_code


def write_all(keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    """Store every document and keep each one's update as its eligibility trace."""
    matrix = values.new_zeros(values.shape[1], keys.shape[1])
    tags = []
    for key, value in zip(keys, values, strict=True):
        delta = torch.outer(value - matrix @ key, key)
        tags.append(delta)
        matrix = matrix + delta
    return matrix, torch.stack(tags)


def coincidence(matrix: Tensor, tags: Tensor, key: Tensor, values: Tensor) -> Tensor:
    """How strongly each stored trace answers a probe, without reading content.

    The modulatory pathway never looks up which document it is addressing. It
    sees only how much each still-eligible trace responds to the probe, which is
    the coincidence the circuit uses to decide where plasticity lands.
    """
    # The response magnitude is the coincidence: a trace written on a key close
    # to the probe answers strongly, one written elsewhere barely answers at all.
    # Normalising it away would leave only a direction, which every trace shares,
    # and the pathway would lose its address.
    response = torch.einsum("tij,j->ti", tags, key)
    return (response * F.normalize(values, dim=1)).sum(dim=1)


def opponent_modulate(
    matrix: Tensor,
    tags: Tensor,
    values: Tensor,
    probe_key: Tensor,
    *,
    sign: float,
    capture: float,
    prediction_error: bool,
) -> Tensor:
    """Apply a signed reinforcement through an approach and an avoidance channel.

    The drive is what the reinforcement asks for minus what the state already
    delivers, so an event the memory has already absorbed releases nothing. The
    two channels carry the positive and negative parts of that residual. A single
    positive channel could only rescale the state, which cannot reorder what it
    retrieves; a difference of two can.
    """
    current = coincidence(matrix, tags, probe_key, values)
    weights = current.clamp(min=0.0)
    wanted = torch.full_like(current, 1.0 if sign > 0 else 0.0)
    # The requested direction is already carried by wanted minus current; the
    # clamping only splits that residual into the two channels, whose difference
    # is what reaches the state.
    drive = (wanted - current) if prediction_error else torch.full_like(current, sign)
    approach = drive.clamp(min=0.0)
    avoid = (-drive).clamp(min=0.0)
    gain = capture * weights * (approach - avoid)
    return matrix + torch.einsum("t,tij->ij", gain, tags)


def recall(matrix: Tensor, keys: Tensor, values: Tensor) -> Tensor:
    read = keys @ matrix.T
    return (F.normalize(read, dim=1) * F.normalize(values, dim=1)).sum(dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Can a content-free opponent pair strengthen one trace and weaken another?"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--key-dim", type=int, default=512)
    parser.add_argument("--active", type=int, default=32)
    parser.add_argument("--load", type=int, default=16)
    parser.add_argument("--capture", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    documents = F.normalize(payload["candidates"].flatten(0, 1), dim=-1)
    feature_dim = documents.shape[-1]

    conditions = {
        "strengthen": (1.0, True, True),
        "weaken": (-1.0, True, True),
        "strengthen_mismatched_probe": (1.0, False, True),
        "weaken_mismatched_probe": (-1.0, False, True),
        "strengthen_without_prediction_error": (1.0, True, False),
    }
    collected: dict[str, list[float]] = {name: [] for name in conditions}
    collected["addressed_share"] = []

    for seed in (int(value) for value in args.seeds.split(",")):
        generator = torch.Generator().manual_seed(seed)
        key_projection = torch.randn(
            feature_dim, args.key_dim, generator=generator
        ) / args.key_dim**0.5
        value_projection = torch.randn(
            feature_dim, args.value_dim, generator=generator
        ) / args.value_dim**0.5
        for _ in range(args.episodes):
            chosen = torch.randperm(len(documents), generator=generator)[: args.load + 1]
            block = documents[chosen]
            keys = sparse_code(torch.tanh(block @ key_projection), args.active)
            values = F.normalize(torch.tanh(block @ value_projection), dim=1)
            stored_keys, stored_values = keys[: args.load], values[: args.load]
            matrix, tags = write_all(stored_keys, stored_values)
            before = recall(matrix, stored_keys, stored_values)
            target = int(
                torch.randint(args.load, (1,), generator=generator)
            )
            weights = coincidence(matrix, tags, stored_keys[target], stored_values)
            collected["addressed_share"].append(
                float(weights.argmax() == target)
            )
            for name, (sign, matched, error) in conditions.items():
                probe = stored_keys[target] if matched else keys[args.load]
                updated = opponent_modulate(
                    matrix,
                    tags,
                    stored_values,
                    probe,
                    sign=sign,
                    capture=args.capture,
                    prediction_error=error,
                )
                after = recall(updated, stored_keys, stored_values)
                collected[name].append(float(after[target] - before[target]))

    output = {
        "config": vars(args),
        "documents": len(documents),
        "addressed_by_coincidence": sum(collected["addressed_share"])
        / len(collected["addressed_share"]),
        "change_in_target_recall": {
            name: sum(values) / len(values)
            for name, values in collected.items()
            if name != "addressed_share"
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
