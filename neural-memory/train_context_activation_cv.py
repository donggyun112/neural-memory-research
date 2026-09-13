from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from neural_memory.context_memory import (
    DualEncoderRecallMemory,
    MultiTraceRecallMemory,
    RecallMode,
)


RecallModel = MultiTraceRecallMemory | DualEncoderRecallMemory


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    top1_hit: float
    episode_exact: float


def measure(logits: Tensor, targets: Tensor, masks: Tensor) -> Metrics:
    predictions = logits > 0
    valid_predictions = predictions[masks]
    valid_targets = targets[masks]
    positive_recall = (valid_predictions[valid_targets]).float().mean()
    negative_recall = (~valid_predictions[~valid_targets]).float().mean()
    true_positive = (valid_predictions & valid_targets).sum()
    predicted_positive = valid_predictions.sum().clamp_min(1)
    masked_logits = logits.masked_fill(~masks, -torch.inf)
    top1 = masked_logits.argmax(dim=-1)
    top1_hit = targets.gather(1, top1[:, None]).float().mean()
    correct = (predictions == targets) | ~masks
    return Metrics(
        accuracy=float((valid_predictions == valid_targets).float().mean()),
        balanced_accuracy=float((positive_recall + negative_recall) / 2),
        precision=float(true_positive / predicted_positive),
        recall=float(positive_recall),
        top1_hit=float(top1_hit),
        episode_exact=float(correct.all(dim=-1).float().mean()),
    )


def shuffle_within_projects(queries: Tensor, project_ids: Tensor) -> Tensor:
    """Pair each trace set with another task query from the same held-out repository."""
    shuffled = queries.clone()
    for project in project_ids.unique():
        indices = (project_ids == project).nonzero(as_tuple=False).flatten()
        if len(indices) > 1:
            shuffled[indices] = queries[indices.roll(1)]
    return shuffled


def transform_features(
    candidates: Tensor,
    queries: Tensor,
    masks: Tensor,
    train_mask: Tensor,
    *,
    mode: str,
) -> tuple[Tensor, Tensor]:
    """Fit an unsupervised feature transform on training projects only."""
    if mode == "none":
        return candidates, queries
    train_candidates = candidates[train_mask][masks[train_mask]]
    train_queries = queries[train_mask]
    observations = torch.cat((train_candidates, train_queries), dim=0)
    mean = observations.mean(dim=0)
    centered_candidates = candidates - mean
    centered_queries = queries - mean
    if mode == "zscore-l2":
        scale = observations.std(dim=0, unbiased=False).clamp_min(1e-5)
        centered_candidates = centered_candidates / scale
        centered_queries = centered_queries / scale
    elif mode != "center-l2":
        raise ValueError(f"unknown feature transform: {mode}")
    transformed_candidates = F.normalize(centered_candidates, dim=-1)
    transformed_candidates = transformed_candidates.masked_fill(~masks[..., None], 0.0)
    return transformed_candidates, F.normalize(centered_queries, dim=-1)


def fit(
    candidates: Tensor,
    queries: Tensor,
    targets: Tensor,
    masks: Tensor,
    *,
    mode: RecallMode,
    steps: int,
    batch_size: int,
    memory_dim: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    architecture: str = "relation",
    alignment_weight: float = 0.0,
) -> RecallModel:
    torch.manual_seed(seed)
    if architecture == "relation":
        model: RecallModel = MultiTraceRecallMemory(candidates.shape[-1], memory_dim).to(device)
    elif architecture == "dual":
        model = DualEncoderRecallMemory(candidates.shape[-1], memory_dim).to(device)
    else:
        raise ValueError(f"unknown architecture: {architecture}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    candidates, queries, targets, masks = (
        tensor.to(device) for tensor in (candidates, queries, targets, masks)
    )
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    model.train()
    for _ in range(steps):
        indices = torch.randint(
            len(targets), (batch_size,), generator=generator, device=device
        )
        batch_targets = targets[indices]
        batch_masks = masks[indices]
        batch_candidates = candidates[indices]
        batch_queries = queries[indices]
        logits = model(batch_candidates, batch_queries, mode=mode)
        loss = F.binary_cross_entropy_with_logits(
            logits[batch_masks], batch_targets[batch_masks].float()
        )
        if alignment_weight > 0 and mode == "joint":
            if not isinstance(model, MultiTraceRecallMemory):
                raise ValueError("alignment loss requires the relation architecture")
            state = model.observe(batch_candidates)
            alignment = model.alignment_logits(state, batch_queries)
            loss = loss + alignment_weight * F.binary_cross_entropy_with_logits(
                alignment[batch_masks], batch_targets[batch_masks].float()
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped-project ContextBench activation CV")
    parser.add_argument("--features", type=Path, default=Path("artifacts/contextbench.pt"))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--architecture", choices=("relation", "dual"), default="relation")
    parser.add_argument(
        "--feature-transform",
        choices=("none", "center-l2", "zscore-l2"),
        default="none",
    )
    parser.add_argument("--alignment-weight", type=float, default=0.0)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.alignment_weight < 0:
        raise ValueError("alignment weight must be non-negative")
    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    candidates = payload["all_candidates"]
    queries = payload["all_queries"]
    targets = payload["all_targets"]
    masks = payload["all_masks"]
    project_ids = payload["all_project_ids"]
    projects = project_ids.unique(sorted=True)
    if args.folds < 2 or args.folds > len(projects):
        raise ValueError("folds must be between 2 and the number of projects")
    fold_projects = [[] for _ in range(args.folds)]
    fold_sizes = [0] * args.folds
    sizes = [
        (int(project), int((project_ids == project).sum())) for project in projects
    ]
    for project, size in sorted(sizes, key=lambda item: (-item[1], item[0])):
        fold = min(range(args.folds), key=lambda index: fold_sizes[index])
        fold_projects[fold].append(project)
        fold_sizes[fold] += size

    train_modes = ("candidate", "query", "joint") if args.architecture == "relation" else ("joint",)
    predictions = {mode: [] for mode in (*train_modes, "joint_shuffled_query")}
    heldout_targets: list[Tensor] = []
    heldout_masks: list[Tensor] = []
    device = torch.device(args.device)
    for fold, heldout in enumerate(fold_projects):
        eval_mask = torch.zeros_like(project_ids, dtype=torch.bool)
        for project in heldout:
            eval_mask |= project_ids == project
        train_mask = ~eval_mask
        fold_candidates, fold_queries = transform_features(
            candidates,
            queries,
            masks,
            train_mask,
            mode=args.feature_transform,
        )
        models: dict[str, RecallModel] = {}
        for mode in train_modes:
            model = fit(
                fold_candidates[train_mask],
                fold_queries[train_mask],
                targets[train_mask],
                masks[train_mask],
                mode=mode,  # type: ignore[arg-type]
                steps=args.steps,
                batch_size=args.batch_size,
                memory_dim=args.memory_dim,
                learning_rate=args.learning_rate,
                seed=args.seed + fold * 1009,
                device=device,
                architecture=args.architecture,
                alignment_weight=args.alignment_weight,
            )
            models[mode] = model
            with torch.no_grad():
                logits = model(
                    fold_candidates[eval_mask].to(device),
                    fold_queries[eval_mask].to(device),
                    mode=mode,  # type: ignore[arg-type]
                )
            predictions[mode].append(logits.cpu())
        shuffled = shuffle_within_projects(fold_queries[eval_mask], project_ids[eval_mask])
        with torch.no_grad():
            logits = models["joint"](
                fold_candidates[eval_mask].to(device), shuffled.to(device), mode="joint"
            )
        predictions["joint_shuffled_query"].append(logits.cpu())
        heldout_targets.append(targets[eval_mask])
        heldout_masks.append(masks[eval_mask])

    combined_targets = torch.cat(heldout_targets)
    combined_masks = torch.cat(heldout_masks)
    results = {
        name: measure(torch.cat(values), combined_targets, combined_masks)
        for name, values in predictions.items()
    }
    output = {
        "config": vars(args),
        "episodes": len(combined_targets),
        "projects": len(projects),
        "labels": int(combined_masks.sum()),
        "metrics": {name: asdict(value) for name, value in results.items()},
    }
    if args.summary:
        fields = tuple(Metrics.__dataclass_fields__)
        print("fields=" + ",".join(fields))
        print(
            ";".join(
                name + "=" + ",".join(f"{getattr(value, field):.6f}" for field in fields)
                for name, value in results.items()
            )
        )
    else:
        print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
