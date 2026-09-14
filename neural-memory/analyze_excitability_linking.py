from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from analyze_sparse_capacity import sparse_code


def excitable_codes(
    projected: Tensor, active: int, *, gain: float, decay: float
) -> tuple[Tensor, Tensor]:
    """Sparsify each document while recently used units stay easier to recruit.

    Excitability is raised by writing and fades afterwards, so documents written
    close together compete for the same units and end up sharing substrate. That
    overlap is the substrate-level basis for linking, as distinct from the
    interference that any two stored items cause each other.

    Returns the codes and the mean overlap between consecutive codes.
    """
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must lie inside [0, 1]")
    excitability = torch.zeros(projected.shape[1])
    codes = []
    for row in projected:
        biased = row + gain * excitability
        code = sparse_code(biased[None], active)[0]
        codes.append(code)
        excitability = decay * excitability + (code != 0).to(excitability.dtype)
    stacked = torch.stack(codes)
    occupied = (stacked != 0).float()
    consecutive = (occupied[:-1] * occupied[1:]).sum(dim=1) / active
    return stacked, consecutive


def cross_recall(matrix: Tensor, keys: Tensor, values: Tensor) -> Tensor:
    """Agreement between what each probe returns and every stored value."""
    read = F.normalize(keys @ matrix.T, dim=1)
    return read @ F.normalize(values, dim=1).T


def by_separation(agreement: Tensor, span: int) -> dict[int, float]:
    """Average off-diagonal agreement at each write-order distance."""
    count = len(agreement)
    result: dict[int, float] = {}
    for distance in range(1, min(span, count - 1) + 1):
        rows = torch.arange(count - distance)
        result[distance] = float(
            torch.cat(
                (agreement[rows, rows + distance], agreement[rows + distance, rows])
            ).mean()
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Do documents written close together end up linked?"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--key-dim", type=int, default=512)
    parser.add_argument("--active", type=int, default=32)
    parser.add_argument("--load", type=int, default=24)
    parser.add_argument("--gains", default="0.0,0.05,0.15")
    parser.add_argument("--excitability-decay", type=float, default=0.5)
    parser.add_argument("--span", type=int, default=6)
    parser.add_argument(
        "--hebbian", action="store_true",
        help="store without error correction, so overlap is not subtracted away",
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    documents = F.normalize(payload["candidates"].flatten(0, 1), dim=-1)
    feature_dim = documents.shape[-1]
    gains = [float(value) for value in args.gains.split(",")]

    results: dict[str, dict[str, float]] = {}
    overlaps: dict[str, float] = {}
    for gain in gains:
        separations: dict[int, list[float]] = {}
        shared: list[float] = []
        for seed in (int(value) for value in args.seeds.split(",")):
            generator = torch.Generator().manual_seed(seed)
            key_projection = torch.randn(
                feature_dim, args.key_dim, generator=generator
            ) / args.key_dim**0.5
            value_projection = torch.randn(
                feature_dim, args.value_dim, generator=generator
            ) / args.value_dim**0.5
            for _ in range(args.episodes):
                chosen = torch.randperm(len(documents), generator=generator)[: args.load]
                block = documents[chosen]
                keys, consecutive = excitable_codes(
                    torch.tanh(block @ key_projection),
                    args.active,
                    gain=gain,
                    decay=args.excitability_decay,
                )
                values = F.normalize(torch.tanh(block @ value_projection), dim=1)
                matrix = values.new_zeros(args.value_dim, args.key_dim)
                for key, value in zip(keys, values, strict=True):
                    # Error correction subtracts what the state already returns
                    # for this key, which is exactly the neighbour's contribution
                    # when the two share substrate. Hebbian storage does not, so
                    # the write rule decides whether overlap links or anti-links.
                    target = value if args.hebbian else value - matrix @ key
                    matrix = matrix + torch.outer(target, key)
                shared.append(float(consecutive.mean()))
                for distance, score in by_separation(
                    cross_recall(matrix, keys, values), args.span
                ).items():
                    separations.setdefault(distance, []).append(score)
        results[str(gain)] = {
            str(distance): sum(scores) / len(scores)
            for distance, scores in sorted(separations.items())
        }
        overlaps[str(gain)] = sum(shared) / len(shared)

    output = {
        "config": vars(args),
        "documents": len(documents),
        "consecutive_code_overlap": overlaps,
        "cross_recall_by_separation": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
