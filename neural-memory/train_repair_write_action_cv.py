from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from neural_memory.repair_memory import (
    MultiTraceActionMemory,
    RepairBanks,
    SelectiveWriteActionMemory,
)
from train_context_activation_cv import measure as binary_measure
from train_context_activation_cv import shuffle_within_projects
from train_context_write_recall_cv import balanced_project_folds
from train_repair_credit import CreditMetrics, measure, train_variant


def train_selective_action(
    banks: RepairBanks,
    write_context: Tensor,
    *,
    keep_ratio: float,
    write_steps: int,
    action_steps: int,
    batch_size: int,
    memory_dim: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> SelectiveWriteActionMemory:
    torch.manual_seed(seed)
    model = SelectiveWriteActionMemory(
        banks.feature_dim, memory_dim, keep_ratio=keep_ratio
    ).to(device)
    candidates, outcomes, targets, masks = (
        tensor.to(device) for tensor in banks.split("train")
    )
    write_context = write_context.to(device)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    write_optimizer = torch.optim.AdamW(
        model.writer.parameters(), lr=learning_rate, weight_decay=1e-3
    )
    active = targets[masks] != 0
    positive_weight = len(active) / (2 * active.sum().clamp_min(1))
    negative_weight = len(active) / (2 * (~active).sum().clamp_min(1))
    model.train()
    for _ in range(write_steps):
        indices = torch.randint(
            len(targets), (batch_size,), generator=generator, device=device
        )
        batch_masks = masks[indices]
        state = model.write(candidates[indices], write_context[indices], batch_masks)
        batch_targets = targets[indices][batch_masks] != 0
        sample_weights = torch.where(
            batch_targets, positive_weight, negative_weight
        )
        loss = F.binary_cross_entropy_with_logits(
            state.gate_logits[batch_masks], batch_targets.float(), weight=sample_weights
        )
        write_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        write_optimizer.step()
    model.writer.requires_grad_(False)
    action_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        action_parameters, lr=learning_rate, weight_decay=1e-3
    )
    selected_targets = targets[masks & (targets != 0)] - 1
    counts = torch.bincount(selected_targets, minlength=2).float()
    weights = counts.sum() / counts.clamp_min(1)
    weights /= weights.mean()
    for _ in range(action_steps):
        indices = torch.randint(
            len(targets), (batch_size,), generator=generator, device=device
        )
        logits, state = model(
            candidates[indices],
            write_context[indices],
            outcomes[indices],
            masks[indices],
        )
        action_mask = masks[indices] & (targets[indices] != 0) & state.selected
        loss = F.cross_entropy(
            logits[action_mask][:, 1:], targets[indices][action_mask] - 1, weight=weights
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Real delayed write/update action CV")
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/repair-write-recall-gemma.pt")
    )
    parser.add_argument("--write-steps", type=int, default=1000)
    parser.add_argument("--action-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--keep-ratio", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    candidates = payload["all_candidates"]
    outcomes = payload["all_recall_queries"]
    write_context = payload["all_write_queries"]
    targets = payload["all_action_targets"]
    masks = payload["all_masks"]
    project_ids = payload["all_project_ids"]
    fold_projects = balanced_project_folds(project_ids, args.folds)
    names = ["selective", "shuffled_outcome", "write_selection"]
    if not args.skip_baseline:
        names.extend(("all_write", "all_write_shuffled"))
    predictions: dict[str, list[Tensor]] = {name: [] for name in names}
    heldout_targets: list[Tensor] = []
    heldout_masks: list[Tensor] = []
    device = torch.device(args.device)

    for fold, heldout in enumerate(fold_projects):
        eval_mask = torch.zeros_like(project_ids, dtype=torch.bool)
        for project in heldout:
            eval_mask |= project_ids == project
        train_mask = ~eval_mask
        banks = RepairBanks(
            train_candidates=candidates[train_mask],
            train_outcomes=outcomes[train_mask],
            train_targets=targets[train_mask],
            train_masks=masks[train_mask],
            eval_candidates=candidates[eval_mask],
            eval_outcomes=outcomes[eval_mask],
            eval_targets=targets[eval_mask],
            eval_masks=masks[eval_mask],
        )
        fold_seed = args.seed + fold * 1009
        model = train_selective_action(
            banks,
            write_context[train_mask],
            keep_ratio=args.keep_ratio,
            write_steps=args.write_steps,
            action_steps=args.action_steps,
            batch_size=args.batch_size,
            memory_dim=args.memory_dim,
            learning_rate=args.learning_rate,
            seed=fold_seed,
            device=device,
        )
        baseline: MultiTraceActionMemory | None = None
        if not args.skip_baseline:
            baseline = train_variant(
                banks,
                mode="joint",
                steps=args.action_steps,
                batch_size=args.batch_size,
                memory_dim=args.memory_dim,
                learning_rate=args.learning_rate,
                seed=fold_seed,
                device=device,
            )
        eval_candidates = banks.eval_candidates.to(device)
        eval_outcomes = banks.eval_outcomes
        eval_masks = banks.eval_masks.to(device)
        eval_write = write_context[eval_mask].to(device)
        eval_projects = project_ids[eval_mask]
        shuffled = shuffle_within_projects(eval_outcomes, eval_projects)
        model.eval()
        with torch.inference_mode():
            logits, state = model(
                eval_candidates, eval_write, eval_outcomes.to(device), eval_masks
            )
            predictions["selective"].append(logits.cpu())
            binary = torch.where(state.selected, 8.0, -8.0)
            predictions["write_selection"].append(binary.cpu())
            logits, _ = model(
                eval_candidates, eval_write, shuffled.to(device), eval_masks
            )
            predictions["shuffled_outcome"].append(logits.cpu())
            if baseline is not None:
                predictions["all_write"].append(
                    baseline(eval_candidates, eval_outcomes.to(device), mode="joint").cpu()
                )
                predictions["all_write_shuffled"].append(
                    baseline(eval_candidates, shuffled.to(device), mode="joint").cpu()
                )
        heldout_targets.append(banks.eval_targets)
        heldout_masks.append(banks.eval_masks)

    combined_targets = torch.cat(heldout_targets)
    combined_masks = torch.cat(heldout_masks)
    results: dict[str, CreditMetrics | object] = {}
    for name, rows in predictions.items():
        logits = torch.cat(rows)
        if name == "write_selection":
            results[name] = binary_measure(
                logits, combined_targets != 0, combined_masks
            )
        else:
            results[name] = measure(logits, combined_targets, combined_masks)
    output = {
        "config": vars(args),
        "episodes": len(combined_targets),
        "projects": int(project_ids.unique().numel()),
        "labels": int(combined_masks.sum()),
        "metrics": {name: asdict(value) for name, value in results.items()},
    }
    if args.summary:
        print(json.dumps(output["metrics"], sort_keys=True))
    else:
        print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
