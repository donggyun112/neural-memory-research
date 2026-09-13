from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from neural_memory.context_memory import SelectiveWriteRecallMemory
from neural_memory.global_feedback import load_global_feedback_banks
from train_context_activation_cv import measure as binary_measure
from train_global_feedback import metrics as ranking_metrics


@dataclass(frozen=True)
class RevisitResult:
    retained_target_rate: float
    position_macro_retention: float
    write_balanced_accuracy: float
    top1: float
    position_macro_top1: float
    conditional_top1: float
    top2: float
    mrr: float


def add_causal_write_features(
    candidates: Tensor,
    feedback: Tensor,
    *,
    ema_decay: float = 0.9,
    feature_mode: str = "all",
) -> tuple[Tensor, Tensor]:
    """Append online-only recency, novelty, and prediction-surprise signals.

    Every feature at slot ``t`` is computed from slots ``<= t``.  The later
    recall query is padded with zeros, so no future information enters write.
    """
    if candidates.ndim != 3 or feedback.ndim != 2:
        raise ValueError("candidates and feedback must be rank 3 and 2")
    if candidates.shape[0] != feedback.shape[0]:
        raise ValueError("candidate and feedback batches must match")
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be inside [0, 1)")
    batch, traces, _ = candidates.shape
    normalized = F.normalize(candidates, dim=-1)
    position = torch.linspace(
        0.0, 1.0, traces, device=candidates.device, dtype=candidates.dtype
    ).view(1, traces, 1).expand(batch, -1, -1)
    novelty = torch.ones(batch, traces, 1, device=candidates.device, dtype=candidates.dtype)
    surprise = torch.ones_like(novelty)
    ema = normalized[:, 0]
    for slot in range(1, traces):
        prior = normalized[:, :slot]
        similarity = (prior * normalized[:, slot : slot + 1]).sum(dim=-1)
        novelty[:, slot, 0] = 0.5 * (1.0 - similarity.max(dim=1).values)
        surprise[:, slot, 0] = 0.5 * (
            1.0 - (normalized[:, slot] * F.normalize(ema, dim=-1)).sum(dim=-1)
        )
        ema = ema_decay * ema + (1.0 - ema_decay) * normalized[:, slot]
    feature_sets = {
        "position": (position,),
        "novelty": (novelty,),
        "surprise": (surprise,),
        "novelty_surprise": (novelty, surprise),
        "all": (position, novelty, surprise),
    }
    if feature_mode not in feature_sets:
        raise ValueError(f"unknown causal feature mode: {feature_mode}")
    additions = feature_sets[feature_mode]
    augmented_candidates = torch.cat((candidates, *additions), dim=-1)
    augmented_feedback = F.pad(feedback, (0, len(additions)))
    return augmented_candidates, augmented_feedback


def write_supervision_loss(
    gate_logits: Tensor,
    teacher_targets: Tensor,
    masks: Tensor,
    *,
    objective: str,
) -> Tensor:
    if objective == "listwise":
        return F.cross_entropy(gate_logits, teacher_targets.float().argmax(dim=1))
    if objective != "bce":
        raise ValueError(f"unknown write objective: {objective}")
    positives = teacher_targets.sum()
    negatives = masks.sum() - positives
    positive_weight = masks.sum() / (2 * positives.clamp_min(1))
    negative_weight = masks.sum() / (2 * negatives.clamp_min(1))
    weights = torch.where(teacher_targets, positive_weight, negative_weight)
    return F.binary_cross_entropy_with_logits(
        gate_logits, teacher_targets.float(), weight=weights
    )


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
    shuffle_write_teacher: bool,
    write_objective: str = "bce",
    causal_features: bool = False,
    causal_feature_mode: str = "all",
    joint_steps: int = 0,
    joint_learning_rate: float = 2e-4,
    joint_teacher_weight: float = 0.25,
    device: torch.device,
) -> SelectiveWriteRecallMemory:
    torch.manual_seed(seed)
    if causal_features:
        candidates, feedback = add_causal_write_features(
            candidates, feedback, feature_mode=causal_feature_mode
        )
    model = SelectiveWriteRecallMemory(
        candidates.shape[-1], memory_dim, keep_ratio=keep_ratio
    ).to(device)
    candidates, feedback, targets = (
        tensor.to(device) for tensor in (candidates, feedback, targets)
    )
    masks = torch.ones(candidates.shape[:2], dtype=torch.bool, device=device)
    write_context = torch.zeros_like(feedback)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    target_matrix = F.one_hot(targets, num_classes=candidates.shape[1]).bool()
    teacher_targets = target_matrix
    if shuffle_write_teacher:
        random_targets = torch.randint(
            candidates.shape[1], targets.shape, generator=generator, device=device
        )
        teacher_targets = F.one_hot(
            random_targets, num_classes=candidates.shape[1]
        ).bool()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-3
    )
    model.train()
    for _ in range(write_steps):
        indices = torch.randint(
            len(targets), (batch_size,), generator=generator, device=device
        )
        state = model.write(candidates[indices], write_context[indices], masks[indices])
        batch_targets = teacher_targets[indices]
        loss = write_supervision_loss(
            state.gate_logits,
            batch_targets,
            masks[indices],
            objective=write_objective,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.candidate_projection.requires_grad_(False)
    model.write_projection.requires_grad_(False)
    model.write_gate.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-3,
    )
    for _ in range(recall_steps):
        indices = torch.randint(
            len(targets), (batch_size,), generator=generator, device=device
        )
        logits, state = model(
            candidates[indices],
            write_context[indices],
            feedback[indices],
            masks[indices],
        )
        retained = state.selected.gather(1, targets[indices][:, None]).squeeze(1)
        if not bool(retained.any()):
            continue
        loss = F.cross_entropy(logits[retained], targets[indices][retained])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if joint_steps:
        model.candidate_projection.requires_grad_(True)
        model.write_projection.requires_grad_(True)
        model.write_gate.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=joint_learning_rate, weight_decay=1e-3
        )
        for _ in range(joint_steps):
            indices = torch.randint(
                len(targets), (batch_size,), generator=generator, device=device
            )
            logits, state = model(
                candidates[indices],
                write_context[indices],
                feedback[indices],
                masks[indices],
            )
            recall_loss = F.cross_entropy(logits, targets[indices])
            teacher_loss = write_supervision_loss(
                state.gate_logits,
                teacher_targets[indices],
                masks[indices],
                objective=write_objective,
            )
            loss = recall_loss + joint_teacher_weight * teacher_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def evaluate(
    model: SelectiveWriteRecallMemory,
    candidates: Tensor,
    feedback: Tensor,
    targets: Tensor,
    *,
    causal_features: bool = False,
    causal_feature_mode: str = "all",
    device: torch.device,
) -> RevisitResult:
    if causal_features:
        candidates, feedback = add_causal_write_features(
            candidates, feedback, feature_mode=causal_feature_mode
        )
    candidates, feedback, targets = (
        tensor.to(device) for tensor in (candidates, feedback, targets)
    )
    masks = torch.ones(candidates.shape[:2], dtype=torch.bool, device=device)
    write_context = torch.zeros_like(feedback)
    model.eval()
    logits, state = model(candidates, write_context, feedback, masks)
    target_matrix = F.one_hot(targets, num_classes=candidates.shape[1]).bool()
    selection_logits = torch.where(state.selected, 8.0, -8.0)
    write_metrics = binary_measure(selection_logits, target_matrix, masks)
    rank = ranking_metrics(logits, targets)
    retained = state.selected.gather(1, targets[:, None]).float().mean()
    retained_mask = state.selected.gather(1, targets[:, None]).squeeze(1)
    predicted = logits.argmax(dim=-1)
    position_retention: list[Tensor] = []
    position_top1: list[Tensor] = []
    for position in range(candidates.shape[1]):
        position_mask = targets == position
        if bool(position_mask.any()):
            position_retention.append(retained_mask[position_mask].float().mean())
            position_top1.append((predicted[position_mask] == targets[position_mask]).float().mean())
    conditional_top1 = float("nan")
    if bool(retained_mask.any()):
        conditional_top1 = float(
            (
                logits.argmax(dim=-1)[retained_mask] == targets[retained_mask]
            ).float().mean().cpu()
        )
    return RevisitResult(
        retained_target_rate=float(retained.cpu()),
        position_macro_retention=float(torch.stack(position_retention).mean().cpu()),
        write_balanced_accuracy=write_metrics.balanced_accuracy,
        top1=rank.top1,
        position_macro_top1=float(torch.stack(position_top1).mean().cpu()),
        conditional_top1=conditional_top1,
        top2=rank.top2,
        mrr=rank.mean_reciprocal_rank,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write-before-revisit natural conversation test")
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/claude_global_feedback.pt")
    )
    parser.add_argument("--write-steps", type=int, default=1000)
    parser.add_argument("--recall-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--keep-ratio", type=float, default=0.75)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shuffle-write-teacher", action="store_true")
    parser.add_argument("--write-objective", choices=("bce", "listwise"), default="bce")
    parser.add_argument("--causal-features", action="store_true")
    parser.add_argument(
        "--causal-feature-mode",
        choices=("position", "novelty", "surprise", "novelty_surprise", "all"),
        default="all",
    )
    parser.add_argument("--joint-steps", type=int, default=0)
    parser.add_argument("--joint-learning-rate", type=float, default=2e-4)
    parser.add_argument("--joint-teacher-weight", type=float, default=0.25)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    banks = load_global_feedback_banks(args.features)
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
        seed=args.seed,
        shuffle_write_teacher=args.shuffle_write_teacher,
        write_objective=args.write_objective,
        causal_features=args.causal_features,
        causal_feature_mode=args.causal_feature_mode,
        joint_steps=args.joint_steps,
        joint_learning_rate=args.joint_learning_rate,
        joint_teacher_weight=args.joint_teacher_weight,
        device=torch.device(args.device),
    )
    correct = evaluate(
        model,
        banks.eval_candidates,
        banks.eval_feedback,
        banks.eval_targets,
        causal_features=args.causal_features,
        causal_feature_mode=args.causal_feature_mode,
        device=torch.device(args.device),
    )
    shuffled = evaluate(
        model,
        banks.eval_candidates,
        banks.eval_feedback.roll(1, dims=0),
        banks.eval_targets,
        causal_features=args.causal_features,
        causal_feature_mode=args.causal_feature_mode,
        device=torch.device(args.device),
    )
    output = {
        "config": vars(args),
        "train_examples": len(banks.train_targets),
        "eval_examples": len(banks.eval_targets),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "correct": asdict(correct),
        "shuffled_recall": asdict(shuffled),
    }
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
