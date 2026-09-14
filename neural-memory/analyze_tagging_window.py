from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from analyze_sparse_capacity import sparse_code


def run_episode(
    keys: Tensor,
    values: Tensor,
    *,
    weak_strength: float,
    intervening: int,
    tag_decay: float,
    capture: float,
    event_strength: float,
) -> float:
    """Write one document weakly, then let a later strong event rescue it.

    Slot 0 is the weakly written target. The next ``intervening`` slots are
    written normally, and the last slot is the strong event.

    A write of strength s expresses s of its update and leaves the unrealised
    ``1 - s`` as a tag, so a full-strength write leaves nothing to capture and
    only the weak trace stands to gain. That asymmetry is the point of the
    mechanism: the strong input is already consolidated and does not need
    rescuing. Tags fade by ``tag_decay`` per subsequent write, and the strong
    event releases a resource that tagged traces capture in proportion to what is
    left of them. The event contributes only magnitude: its content never enters
    the capture term, which is what makes the pathway content-free.

    Returns the cosine between what the state returns for the target's key and
    what was written there.
    """
    if not 0.0 < tag_decay <= 1.0:
        raise ValueError("tag_decay must be inside (0, 1]")
    if intervening + 2 > len(keys):
        raise ValueError("not enough documents for this many intervening writes")
    matrix = values.new_zeros(values.shape[1], keys.shape[1])
    tags: list[Tensor] = []
    order = [0, *range(1, intervening + 1)]
    for position, slot in enumerate(order):
        strength = weak_strength if position == 0 else 1.0
        full = torch.outer(values[slot] - matrix @ keys[slot], keys[slot])
        tags = [tag * tag_decay for tag in tags]
        if strength < 1.0:
            tags.append((1.0 - strength) * full)
        matrix = matrix + strength * full
    event = len(keys) - 1
    matrix = matrix + event_strength * torch.outer(
        values[event] - matrix @ keys[event], keys[event]
    )
    if capture > 0.0 and tags:
        matrix = matrix + capture * event_strength * sum(tags)
    read = matrix @ keys[0]
    return float(F.cosine_similarity(read[None], values[0][None]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Does a later strong event rescue a weak trace, and for how long?"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--key-dim", type=int, default=512)
    parser.add_argument("--active", type=int, default=32)
    parser.add_argument("--weak-strength", type=float, default=0.2)
    parser.add_argument("--event-strength", type=float, default=2.0)
    parser.add_argument("--capture", type=float, default=0.25)
    parser.add_argument("--tag-decays", default="1.0,0.9,0.7,0.5")
    parser.add_argument("--intervening", default="0,1,2,4,8,16")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    documents = F.normalize(payload["candidates"].flatten(0, 1), dim=-1)
    feature_dim = documents.shape[-1]
    gaps = [int(value) for value in args.intervening.split(",")]
    decays = [float(value) for value in args.tag_decays.split(",")]
    width = max(gaps) + 2

    results: dict[str, dict[str, dict[str, float]]] = {}
    for tag_decay in decays:
        rescued: dict[str, list[float]] = {str(gap): [] for gap in gaps}
        control: dict[str, list[float]] = {str(gap): [] for gap in gaps}
        ceiling: dict[str, list[float]] = {str(gap): [] for gap in gaps}
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
                for gap in gaps:
                    shared = {
                        "weak_strength": args.weak_strength,
                        "intervening": gap,
                        "tag_decay": tag_decay,
                        "event_strength": args.event_strength,
                    }
                    rescued[str(gap)].append(
                        run_episode(keys, values, capture=args.capture, **shared)
                    )
                    # The strong event is written either way; only capture differs,
                    # so its extra interference cannot masquerade as rescue.
                    control[str(gap)].append(run_episode(keys, values, capture=0.0, **shared))
                    # What the same trace would have reached had it been written
                    # at full strength. Rescue is reported against this, because
                    # an absolute gain grows simply from the baseline falling.
                    ceiling[str(gap)].append(
                        run_episode(
                            keys,
                            values,
                            capture=0.0,
                            **{**shared, "weak_strength": 1.0},
                        )
                    )
        average = lambda block: {gap: sum(v) / len(v) for gap, v in block.items()}
        with_capture, without, best = average(rescued), average(control), average(ceiling)
        results[str(tag_decay)] = {
            "rescued": with_capture,
            "control": without,
            "ceiling": best,
            "recovered_fraction": {
                gap: (with_capture[gap] - without[gap]) / (best[gap] - without[gap])
                if best[gap] - without[gap] > 1e-6
                else 0.0
                for gap in with_capture
            },
        }

    output = {"config": vars(args), "documents": len(documents), "by_tag_decay": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
