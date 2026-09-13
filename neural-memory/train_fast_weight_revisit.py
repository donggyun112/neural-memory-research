from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev

import torch
from torch import Tensor
from torch.nn import functional as F

from neural_memory.context_memory import FastWeightRecallMemory
from neural_memory.global_feedback import load_global_feedback_banks
from train_context_activation_cv import measure as binary_measure
from train_global_feedback import metrics as ranking_metrics
from train_revisit_write_recall import RevisitResult, write_supervision_loss


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": mean(row[key] for row in rows),
            "std": stdev(row[key] for row in rows) if len(rows) > 1 else 0.0,
        }
        for key in rows[0]
    }


def train_model(
    candidates: Tensor,
    feedback: Tensor,
    targets: Tensor,
    *,
    keep_ratio: float,
    write_steps: int,
    recall_steps: int,
    batch_size: int,
    memory_dim: int,
    learning_rate: float,
    seed: int,
    write_objective: str,
    adaptive_retention: bool,
    device: torch.device,
) -> FastWeightRecallMemory:
    torch.manual_seed(seed)
    model = FastWeightRecallMemory(
        candidates.shape[-1],
        memory_dim,
        keep_ratio=keep_ratio,
        adaptive_retention=adaptive_retention,
    ).to(device)
    candidates, feedback, targets = (
        tensor.to(device) for tensor in (candidates, feedback, targets)
    )
    masks = torch.ones(candidates.shape[:2], dtype=torch.bool, device=device)
    teachers = F.one_hot(targets, num_classes=candidates.shape[1]).bool()
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    model.train()
    for _ in range(write_steps):
        indices = torch.randint(
            len(targets), (batch_size,), generator=generator, device=device
        )
        state = model.write(candidates[indices], masks[indices])
        loss = write_supervision_loss(
            state.gate_logits,
            teachers[indices],
            masks[indices],
            objective=write_objective,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    for module in (model.candidate_key, model.candidate_value, model.write_gate):
        module.requires_grad_(False)
    model.retention_logit.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-3,
    )
    for _ in range(recall_steps):
        indices = torch.randint(
            len(targets), (batch_size,), generator=generator, device=device
        )
        logits, state = model(candidates[indices], feedback[indices], masks[indices])
        retained = state.selected.gather(1, targets[indices, None]).squeeze(1)
        if not bool(retained.any()):
            continue
        loss = F.cross_entropy(logits[retained], targets[indices][retained])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def evaluate(
    model: FastWeightRecallMemory,
    candidates: Tensor,
    feedback: Tensor,
    targets: Tensor,
    *,
    device: torch.device,
) -> RevisitResult:
    candidates, feedback, targets = (
        tensor.to(device) for tensor in (candidates, feedback, targets)
    )
    masks = torch.ones(candidates.shape[:2], dtype=torch.bool, device=device)
    model.eval()
    logits, state = model(candidates, feedback, masks)
    target_matrix = F.one_hot(targets, num_classes=candidates.shape[1]).bool()
    selection_logits = torch.where(state.selected, 8.0, -8.0)
    write_metrics = binary_measure(selection_logits, target_matrix, masks)
    rank = ranking_metrics(logits, targets)
    retained_mask = state.selected.gather(1, targets[:, None]).squeeze(1)
    predicted = logits.argmax(dim=-1)
    position_retention = []
    position_top1 = []
    for position in range(candidates.shape[1]):
        position_mask = targets == position
        if bool(position_mask.any()):
            position_retention.append(retained_mask[position_mask].float().mean())
            position_top1.append((predicted[position_mask] == targets[position_mask]).float().mean())
    conditional = float("nan")
    if bool(retained_mask.any()):
        conditional = float((predicted[retained_mask] == targets[retained_mask]).float().mean())
    return RevisitResult(
        retained_target_rate=float(retained_mask.float().mean()),
        position_macro_retention=float(torch.stack(position_retention).mean()),
        write_balanced_accuracy=write_metrics.balanced_accuracy,
        top1=rank.top1,
        position_macro_top1=float(torch.stack(position_top1).mean()),
        conditional_top1=conditional,
        top2=rank.top2,
        mrr=rank.mean_reciprocal_rank,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Parametric fast-weight revisit memory")
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/claude_global_feedback.pt")
    )
    parser.add_argument("--write-steps", type=int, default=1000)
    parser.add_argument("--recall-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--keep-ratio", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds")
    parser.add_argument("--write-objective", choices=("bce", "listwise"), default="listwise")
    parser.add_argument("--adaptive-retention", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    banks = load_global_feedback_banks(args.features)
    seeds = [args.seed] if args.seeds is None else [int(value) for value in args.seeds.split(",")]
    rows: list[dict[str, float]] = []
    for seed in seeds:
        model = train_model(
            banks.train_candidates,
            banks.train_feedback,
            banks.train_targets,
            keep_ratio=args.keep_ratio,
            write_steps=args.write_steps,
            recall_steps=args.recall_steps,
            batch_size=args.batch_size,
            memory_dim=args.memory_dim,
            learning_rate=args.learning_rate,
            seed=seed,
            write_objective=args.write_objective,
            adaptive_retention=args.adaptive_retention,
            device=device,
        )
        correct = asdict(
            evaluate(
                model,
                banks.eval_candidates,
                banks.eval_feedback,
                banks.eval_targets,
                device=device,
            )
        )
        shuffled = asdict(
            evaluate(
                model,
                banks.eval_candidates,
                banks.eval_feedback.roll(1, dims=0),
                banks.eval_targets,
                device=device,
            )
        )
        row = {f"correct_{key}": value for key, value in correct.items()}
        row.update({f"shuffled_{key}": value for key, value in shuffled.items()})
        row["recall_top1_gap"] = correct["top1"] - shuffled["top1"]
        rows.append(row)
    output = {
        "config": vars(args),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "persistent_state_scalars": args.memory_dim**2,
        "per_seed": rows,
        "summary": summarize(rows),
    }
    serialized = json.dumps(output, indent=2, sort_keys=True, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
        print(f"saved fast-weight results to {args.output}")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
