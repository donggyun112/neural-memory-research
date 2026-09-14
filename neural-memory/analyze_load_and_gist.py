from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from analyze_sparse_capacity import dissimilar_set, similar_set, sparse_code


def store(keys: Tensor, values: Tensor, decay: float) -> Tensor:
    """Delta-rule storage with a multiplicative decay applied at every write."""
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be inside (0, 1]")
    matrix = values.new_zeros(values.shape[1], keys.shape[1])
    for key, value in zip(keys, values, strict=True):
        matrix = decay * matrix + torch.outer(value - matrix @ key, key)
    return matrix


def gist_basis(values: Tensor, rank: int) -> Tensor:
    """The directions the stored documents share, as an orthonormal basis.

    Superposition averages away what is idiosyncratic and reinforces what is
    common, so the shared subspace is the natural stand-in for gist and its
    complement for surface detail.
    """
    if not 1 <= rank < values.shape[1]:
        raise ValueError("rank must be at least one and below the value width")
    if len(values) < rank:
        raise ValueError("need at least rank documents to estimate shared structure")
    # Deliberately uncentred. Centring would make the basis describe how the
    # documents differ, which is the opposite of what is shared between them.
    # Estimate this once from a wide sample, never from the stored set: a
    # fixed-rank basis fitted to the stored documents explains less of each one
    # as the set grows, which moves the split with load and makes the loads
    # incomparable.
    return torch.linalg.svd(values, full_matrices=False).Vh[:rank]


def split_components(values: Tensor, basis: Tensor) -> tuple[Tensor, Tensor]:
    """Decompose each document into its shared and its idiosyncratic part."""
    shared = (values @ basis.T) @ basis
    return shared, values - shared


def recovery(read: Tensor, component: Tensor) -> Tensor:
    """Cosine between a read and one component, ignoring documents with none."""
    norms = component.norm(dim=1)
    usable = norms > 1e-6
    scores = torch.zeros(len(read))
    if bool(usable.any()):
        scores[usable] = (
            F.normalize(read[usable], dim=1) * F.normalize(component[usable], dim=1)
        ).sum(dim=1)
    return scores[usable] if bool(usable.any()) else scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load curve, adaptive decay, and whether surface fades before gist"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--key-dim", type=int, default=512)
    parser.add_argument("--active", type=int, default=32)
    parser.add_argument("--gist-rank", type=int, default=4)
    parser.add_argument("--gist-sample", type=int, default=1000)
    parser.add_argument("--loads", default="4,8,16,32,64,128")
    parser.add_argument("--decays", default="1.0,0.99,0.95,0.9,0.8")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--regime", choices=("similar", "dissimilar"), default="similar")
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    documents = F.normalize(payload["candidates"].flatten(0, 1), dim=-1)
    feature_dim = documents.shape[-1]
    builder = similar_set if args.regime == "similar" else dissimilar_set
    loads = [int(value) for value in args.loads.split(",")]
    decays = [float(value) for value in args.decays.split(",")]

    by_load: dict[str, dict[str, float]] = {}
    for load in loads:
        fidelity: dict[float, list[float]] = {decay: [] for decay in decays}
        gist: list[float] = []
        surface: list[float] = []
        for seed in (int(value) for value in args.seeds.split(",")):
            generator = torch.Generator().manual_seed(seed)
            key_projection = torch.randn(
                feature_dim, args.key_dim, generator=generator
            ) / args.key_dim**0.5
            value_projection = torch.randn(
                feature_dim, args.value_dim, generator=generator
            ) / args.value_dim**0.5
            sample = documents[
                torch.randperm(len(documents), generator=generator)[: args.gist_sample]
            ]
            basis = gist_basis(
                F.normalize(torch.tanh(sample @ value_projection), dim=1), args.gist_rank
            )
            for _ in range(args.episodes):
                anchor = int(torch.randint(len(documents), (1,), generator=generator))
                block = documents[builder(documents, load, anchor)]
                keys = sparse_code(torch.tanh(block @ key_projection), args.active)
                values = F.normalize(torch.tanh(block @ value_projection), dim=1)
                for decay in decays:
                    read = keys @ store(keys, values, decay).T
                    fidelity[decay].append(
                        float(
                            (F.normalize(read, dim=1) * values).sum(dim=1).mean()
                        )
                    )
                read = keys @ store(keys, values, 1.0).T
                shared, residual = split_components(values, basis)
                gist.append(float(recovery(read, shared).mean()))
                surface.append(float(recovery(read, residual).mean()))
        best = max(decays, key=lambda decay: sum(fidelity[decay]) / len(fidelity[decay]))
        by_load[str(load)] = {
            **{
                f"fidelity_decay_{decay:g}": sum(values) / len(values)
                for decay, values in fidelity.items()
            },
            "best_decay": best,
            "gist_recovery": sum(gist) / len(gist),
            "surface_recovery": sum(surface) / len(surface),
        }

    output = {
        "config": vars(args),
        "documents": len(documents),
        "by_load": by_load,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
