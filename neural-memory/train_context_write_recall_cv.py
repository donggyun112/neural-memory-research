from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from neural_memory.context_memory import SelectiveWriteRecallMemory
from train_context_activation_cv import Metrics, fit, measure, shuffle_within_projects


def balanced_project_folds(project_ids: Tensor, folds: int) -> list[list[int]]:
    projects = project_ids.unique(sorted=True)
    if folds < 2 or folds > len(projects):
        raise ValueError("folds must be between 2 and the number of projects")
    groups: list[list[int]] = [[] for _ in range(folds)]
    sizes = [0] * folds
    counts = [
        (int(project), int((project_ids == project).sum())) for project in projects
    ]
    for project, size in sorted(counts, key=lambda item: (-item[1], item[0])):
        fold = min(range(folds), key=lambda index: sizes[index])
        groups[fold].append(project)
        sizes[fold] += size
    return groups


def fit_selective(
    candidates: Tensor,
    write_queries: Tensor,
    recall_queries: Tensor,
    targets: Tensor,
    masks: Tensor,
    *,
    steps: int,
    write_pretrain_steps: int,
    freeze_write: bool,
    batch_size: int,
    memory_dim: int,
    keep_ratio: float,
    learning_rate: float,
    write_aux_weight: float,
    seed: int,
    device: torch.device,
) -> SelectiveWriteRecallMemory:
    torch.manual_seed(seed)
    model = SelectiveWriteRecallMemory(
        candidates.shape[-1], memory_dim, keep_ratio=keep_ratio
    ).to(device)
    candidates, write_queries, recall_queries, targets, masks = (
        tensor.to(device)
        for tensor in (candidates, write_queries, recall_queries, targets, masks)
    )
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    model.train()
    if write_pretrain_steps > 0:
        write_optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-3
        )
        for _ in range(write_pretrain_steps):
            indices = torch.randint(
                len(targets), (batch_size,), generator=generator, device=device
            )
            batch_masks = masks[indices]
            state = model.write(candidates[indices], write_queries[indices], batch_masks)
            loss = F.binary_cross_entropy_with_logits(
                state.gate_logits[batch_masks], targets[indices][batch_masks].float()
            )
            write_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            write_optimizer.step()
    if freeze_write:
        for module in (
            model.candidate_projection,
            model.write_projection,
            model.write_gate,
        ):
            module.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-3,
    )
    for _ in range(steps):
        indices = torch.randint(
            len(targets), (batch_size,), generator=generator, device=device
        )
        batch_masks = masks[indices]
        logits, state = model(
            candidates[indices],
            write_queries[indices],
            recall_queries[indices],
            batch_masks,
        )
        loss = F.binary_cross_entropy_with_logits(
            logits[batch_masks], targets[indices][batch_masks].float()
        )
        if write_aux_weight > 0:
            loss = loss + write_aux_weight * F.binary_cross_entropy_with_logits(
                state.gate_logits[batch_masks], targets[indices][batch_masks].float()
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def hard_selection_metrics(selected: Tensor, targets: Tensor, masks: Tensor) -> Metrics:
    logits = torch.where(selected, 8.0, -8.0)
    return measure(logits, targets, masks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Delayed write-then-recall ContextBench CV")
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/contextbench-gemma-split.pt")
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--write-pretrain-steps", type=int, default=0)
    parser.add_argument("--freeze-write", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--write-aux-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    if args.write_aux_weight < 0:
        raise ValueError("write auxiliary weight must be non-negative")

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    candidates = payload["all_candidates"]
    write_queries = payload["all_write_queries"]
    recall_queries = payload["all_recall_queries"]
    targets = payload["all_targets"]
    masks = payload["all_masks"]
    project_ids = payload["all_project_ids"]
    fold_projects = balanced_project_folds(project_ids, args.folds)
    device = torch.device(args.device)
    result_names = [
        "selective",
        "shuffled_write",
        "shuffled_recall",
        "write_selection",
    ]
    if not args.skip_baselines:
        result_names.extend(("write_only", "recall_only"))
    predictions: dict[str, list[Tensor]] = {name: [] for name in result_names}
    heldout_targets: list[Tensor] = []
    heldout_masks: list[Tensor] = []

    for fold, heldout in enumerate(fold_projects):
        eval_mask = torch.zeros_like(project_ids, dtype=torch.bool)
        for project in heldout:
            eval_mask |= project_ids == project
        train_mask = ~eval_mask
        fold_seed = args.seed + fold * 1009
        selective = fit_selective(
            candidates[train_mask],
            write_queries[train_mask],
            recall_queries[train_mask],
            targets[train_mask],
            masks[train_mask],
            steps=args.steps,
            write_pretrain_steps=args.write_pretrain_steps,
            freeze_write=args.freeze_write,
            batch_size=args.batch_size,
            memory_dim=args.memory_dim,
            keep_ratio=args.keep_ratio,
            learning_rate=args.learning_rate,
            write_aux_weight=args.write_aux_weight,
            seed=fold_seed,
            device=device,
        )
        baselines = {}
        if not args.skip_baselines:
            baselines = {
                "write_only": fit(
                candidates[train_mask],
                write_queries[train_mask],
                targets[train_mask],
                masks[train_mask],
                mode="joint",
                steps=args.steps,
                batch_size=args.batch_size,
                memory_dim=args.memory_dim,
                learning_rate=args.learning_rate,
                seed=fold_seed,
                device=device,
                ),
                "recall_only": fit(
                candidates[train_mask],
                recall_queries[train_mask],
                targets[train_mask],
                masks[train_mask],
                mode="joint",
                steps=args.steps,
                batch_size=args.batch_size,
                memory_dim=args.memory_dim,
                learning_rate=args.learning_rate,
                seed=fold_seed,
                device=device,
                ),
            }
        eval_candidates = candidates[eval_mask].to(device)
        eval_write = write_queries[eval_mask]
        eval_recall = recall_queries[eval_mask]
        eval_masks = masks[eval_mask].to(device)
        eval_projects = project_ids[eval_mask]
        shuffled_write = shuffle_within_projects(eval_write, eval_projects)
        shuffled_recall = shuffle_within_projects(eval_recall, eval_projects)
        selective.eval()
        with torch.inference_mode():
            logits, state = selective(
                eval_candidates,
                eval_write.to(device),
                eval_recall.to(device),
                eval_masks,
            )
            predictions["selective"].append(logits.cpu())
            predictions["write_selection"].append(
                torch.where(state.selected, 8.0, -8.0).cpu()
            )
            logits, _ = selective(
                eval_candidates,
                shuffled_write.to(device),
                eval_recall.to(device),
                eval_masks,
            )
            predictions["shuffled_write"].append(logits.cpu())
            logits, _ = selective(
                eval_candidates,
                eval_write.to(device),
                shuffled_recall.to(device),
                eval_masks,
            )
            predictions["shuffled_recall"].append(logits.cpu())
            for name, model in baselines.items():
                query = eval_write if name == "write_only" else eval_recall
                predictions[name].append(
                    model(eval_candidates, query.to(device), mode="joint").cpu()
                )
        heldout_targets.append(targets[eval_mask])
        heldout_masks.append(masks[eval_mask])

    combined_targets = torch.cat(heldout_targets)
    combined_masks = torch.cat(heldout_masks)
    results = {
        name: measure(torch.cat(logits), combined_targets, combined_masks)
        for name, logits in predictions.items()
    }
    output = {
        "config": vars(args),
        "episodes": len(combined_targets),
        "projects": int(project_ids.unique().numel()),
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
