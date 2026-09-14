from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch import Tensor
from torch.nn import functional as F

from neural_memory.context_memory import DeferredConsolidationMemory
from train_revisit_write_recall import write_supervision_loss


VARIANTS = {
    # provisional_ratio, event mode, how stage two reaches encoder similarity
    "single_stage": (0.25, "correct", "none"),
    "deferred": (0.50, "correct", "none"),
    "deferred_shuffled_event": (0.50, "shuffled", "none"),
    "deferred_blank_event": (0.50, "blank", "none"),
    "deferred_similarity": (0.50, "correct", "feature"),
    "deferred_similarity_shuffled": (0.50, "shuffled", "feature"),
    "deferred_residual": (0.50, "correct", "residual"),
    "deferred_residual_shuffled": (0.50, "shuffled", "residual"),
    "full_deferral_residual": (1.00, "correct", "residual"),
    "full_deferral_residual_shuffled": (1.00, "shuffled", "residual"),
}


def resolve_variants(
    names: str,
    ratios: str,
    modes: str,
    keep_ratio: float,
    similarity: str = "none",
) -> dict[str, tuple[float, str, str]]:
    """Named variants by default, or a provisional-capacity sweep when asked.

    The sweep traces the axis between deciding at write time and deciding once
    the later event exists: at ``keep_ratio`` nothing is deferred, at 1.0 nothing
    is discarded early.
    """
    if not ratios:
        unknown = [name for name in names.split(",") if name not in VARIANTS]
        if unknown:
            raise ValueError(f"unknown variants {unknown}; available: {sorted(VARIANTS)}")
        return {name: VARIANTS[name] for name in names.split(",")}
    resolved: dict[str, tuple[float, str, str]] = {}
    for ratio in ratios.split(","):
        value = float(ratio)
        if not keep_ratio <= value <= 1.0:
            raise ValueError(
                f"provisional ratio {value} must lie between keep_ratio {keep_ratio} and 1.0"
            )
        for mode in modes.split(","):
            resolved[f"provisional{value:g}_{mode}"] = (value, mode, similarity)
    return resolved


def stratified_folds(question_type: Tensor, folds: int) -> Tensor:
    """Assign folds round-robin inside each question type, deterministically."""
    if folds < 2:
        raise ValueError("at least two folds are required")
    assignment = torch.zeros(len(question_type), dtype=torch.long)
    for value in question_type.unique(sorted=True):
        indices = (question_type == value).nonzero().flatten()
        assignment[indices] = torch.arange(len(indices)) % folds
    return assignment


def event_tensor(consolidation: Tensor, mode: str) -> Tensor:
    if mode == "correct":
        return consolidation
    if mode == "shuffled":
        # A real event from another question: same distribution, wrong episode.
        return consolidation.roll(1, dims=0)
    if mode == "blank":
        return torch.zeros_like(consolidation)
    raise ValueError(f"unknown event mode: {mode}")


def train_fold(
    candidates: Tensor,
    event: Tensor,
    query: Tensor,
    targets: Tensor,
    *,
    provisional_ratio: float,
    keep_ratio: float,
    similarity: str,
    center_similarity: bool,
    correction_bound: float,
    consolidation_objective: str,
    memory_dim: int,
    write_steps: int,
    consolidation_steps: int,
    recall_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> DeferredConsolidationMemory:
    torch.manual_seed(seed)
    model = DeferredConsolidationMemory(
        candidates.shape[-1], memory_dim, provisional_ratio, keep_ratio, similarity,
        center_similarity, correction_bound,
    ).to(device)
    masks = torch.ones(candidates.shape[:2], dtype=torch.bool, device=device)
    encoder_similarity = (
        model.encoder_similarity(candidates, event) if similarity != "none" else None
    )

    def similarity_for(rows: Tensor) -> Tensor | None:
        return None if encoder_similarity is None else encoder_similarity[rows]

    teacher = F.one_hot(targets, num_classes=candidates.shape[1]).bool()
    generator = torch.Generator(device=device).manual_seed(seed + 1)

    def batch() -> Tensor:
        return torch.randint(
            len(targets), (batch_size,), generator=generator, device=device
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    model.train()
    for _ in range(write_steps):
        rows = batch()
        state = model.write(candidates[rows], masks[rows])
        loss = write_supervision_loss(
            state.write_logits, teacher[rows], masks[rows], objective="bce"
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.candidate_projection.requires_grad_(False)
    model.write_gate.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-3,
    )
    for _ in range(consolidation_steps):
        rows = batch()
        state = model.write(candidates[rows], masks[rows])
        eligible = state.provisional
        # Only episodes whose target survived the write stage can teach stage two.
        usable = teacher[rows].logical_and(eligible).any(dim=1)
        if not bool(usable.any()):
            continue
        consolidated = model.consolidate(state, event[rows], similarity_for(rows))
        if consolidation_objective == "listwise":
            # Selecting k of n is a ranking problem, and a softmax over the
            # eligible slots is invariant to the constant offset that a positive
            # similarity residual would otherwise force the head to cancel.
            scored = consolidated.consolidation_logits[usable].masked_fill(
                ~eligible[usable], -torch.inf
            )
            loss = F.cross_entropy(scored, targets[rows][usable])
        else:
            loss = write_supervision_loss(
                consolidated.consolidation_logits[usable],
                teacher[rows][usable],
                eligible[usable],
                objective="bce",
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.event_projection.requires_grad_(False)
    model.consolidation_head.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-3,
    )
    for _ in range(recall_steps):
        rows = batch()
        logits, state = model(candidates[rows], event[rows], query[rows], masks[rows])
        survived = state.selected.gather(1, targets[rows][:, None]).squeeze(1)
        if not bool(survived.any()):
            continue
        loss = F.cross_entropy(logits[survived], targets[rows][survived])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def predict(
    model: DeferredConsolidationMemory,
    candidates: Tensor,
    event: Tensor,
    query: Tensor,
    masks: Tensor,
    keep: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    model.eval()
    state = model.write(candidates, masks)
    provisional = state.provisional
    # What a trivial consolidation rule would keep from the same provisional
    # set: this separates a weak second stage from a weak first one.
    similarity = model.encoder_similarity(candidates, event)
    consolidated = model.consolidate(
        state, event, similarity if model.similarity != "none" else None
    )
    logits = model.recall(consolidated, query)
    ceiling = torch.zeros_like(provisional)
    ceiling.scatter_(
        1, similarity.masked_fill(~provisional, -torch.inf).topk(keep, dim=1).indices, True
    )
    return provisional, consolidated.selected, logits.argmax(dim=-1), ceiling


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Does a later related event improve a bounded write decision?"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument(
        "--provisional-ratios",
        default="",
        help="sweep these provisional capacities instead of the named variants",
    )
    parser.add_argument("--event-modes", default="correct,shuffled")
    parser.add_argument(
        "--similarity",
        choices=("none", "feature", "residual"),
        default="none",
        help="how the swept variants reach the frozen-encoder candidate/event similarity",
    )
    parser.add_argument(
        "--consolidation-objective", choices=("bce", "listwise"), default="bce"
    )
    parser.add_argument("--no-center-similarity", action="store_true")
    parser.add_argument(
        "--correction-bound", type=float, default=0.0,
        help="cap the consolidation head's contribution; 0 leaves it unbounded",
    )
    parser.add_argument(
        "--types",
        default="multi-session,temporal-reasoning",
        help=(
            "question types to train and evaluate on; knowledge-update is excluded by "
            "default because its later evidence supersedes the write target and is "
            "never a candidate, so retaining that target is not the right behaviour"
        ),
    )
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--keep-ratio", type=float, default=0.25)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--write-steps", type=int, default=600)
    parser.add_argument("--consolidation-steps", type=int, default=600)
    parser.add_argument("--recall-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    all_names = payload["question_type_names"]
    wanted = [name.strip() for name in args.types.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in all_names]
    if unknown:
        raise ValueError(f"unknown question types {unknown}; available: {all_names}")
    keep_rows = torch.zeros(len(payload["targets"]), dtype=torch.bool)
    for name in wanted:
        keep_rows |= payload["question_type"] == all_names.index(name)
    if not bool(keep_rows.any()):
        raise ValueError("the requested question types select no episodes")
    payload = {
        key: (value[keep_rows] if torch.is_tensor(value) and len(value) == len(keep_rows) else value)
        for key, value in payload.items()
    }
    candidates = payload["candidates"]
    consolidation = payload["consolidation"]
    query = payload["query"]
    targets = payload["targets"]
    lengths = payload["lengths"]
    device = torch.device(args.device)
    episodes, traces, _ = candidates.shape
    keep = max(1, round(traces * args.keep_ratio))

    longest_selection = torch.zeros(episodes, traces, dtype=torch.bool)
    longest_selection.scatter_(1, lengths.topk(keep, dim=1).indices, True)
    longest = float(
        (lengths.topk(keep, dim=1).indices == targets[:, None]).any(dim=1).float().mean()
    )
    cosine = torch.einsum("etf,ef->et", candidates, query)
    query_visible = float(
        (cosine.topk(keep, dim=1).indices == targets[:, None]).any(dim=1).float().mean()
    )
    # The consolidation event is nearly as query-like as the query itself, so
    # this is the reference the deferred variants must be read against.
    event_cosine = torch.einsum("etf,ef->et", candidates, consolidation)
    event_visible = float(
        (event_cosine.topk(keep, dim=1).indices == targets[:, None]).any(dim=1).float().mean()
    )

    type_names = payload["question_type_names"]
    per_type_longest: dict[str, float] = {}
    for index, label in enumerate(type_names):
        subset = payload["question_type"] == index
        if not bool(subset.any()):
            continue
        per_type_longest[label] = float(
            (lengths[subset].topk(keep, dim=1).indices == targets[subset][:, None])
            .any(dim=1)
            .float()
            .mean()
        )
    assignment = stratified_folds(payload["question_type"], args.folds)
    masks = torch.ones(episodes, traces, dtype=torch.bool, device=device)
    seeds = [int(value) for value in args.seeds.split(",")]
    results: dict[str, dict[str, dict[str, float]]] = {}

    plan = resolve_variants(
        args.variants,
        args.provisional_ratios,
        args.event_modes,
        args.keep_ratio,
        args.similarity,
    )
    for name, (provisional_ratio, event_mode, similarity_mode) in plan.items():
        event = event_tensor(consolidation, event_mode)
        rows: list[dict[str, float]] = []
        for seed in seeds:
            kept = torch.zeros(episodes, dtype=torch.bool)
            survived_write = torch.zeros(episodes, dtype=torch.bool)
            correct = torch.zeros(episodes, dtype=torch.bool)
            chosen = torch.zeros(episodes, traces, dtype=torch.bool)
            cosine_kept = torch.zeros(episodes, dtype=torch.bool)
            for fold in range(args.folds):
                evaluate = assignment == fold
                train = ~evaluate
                model = train_fold(
                    candidates[train].to(device),
                    event[train].to(device),
                    query[train].to(device),
                    targets[train].to(device),
                    provisional_ratio=provisional_ratio,
                    keep_ratio=args.keep_ratio,
                    similarity=similarity_mode,
                    center_similarity=not args.no_center_similarity,
                    correction_bound=args.correction_bound,
                    consolidation_objective=args.consolidation_objective,
                    memory_dim=args.memory_dim,
                    write_steps=args.write_steps,
                    consolidation_steps=args.consolidation_steps,
                    recall_steps=args.recall_steps,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=seed,
                    device=device,
                )
                provisional, selected, predicted, ceiling = predict(
                    model,
                    candidates[evaluate].to(device),
                    event[evaluate].to(device),
                    query[evaluate].to(device),
                    masks[: int(evaluate.sum())],
                    keep,
                )
                gold = targets[evaluate].to(device)
                survived_write[evaluate] = provisional.gather(1, gold[:, None]).squeeze(1).cpu()
                kept[evaluate] = selected.gather(1, gold[:, None]).squeeze(1).cpu()
                correct[evaluate] = (predicted == gold).cpu()
                chosen[evaluate] = selected.cpu()
                cosine_kept[evaluate] = ceiling.gather(1, gold[:, None]).squeeze(1).cpu()
            # Equal retention does not mean the same rule: this measures whether
            # the learned writer picks the same slots a length rule would.
            intersection = (chosen & longest_selection).sum(dim=1).float()
            union = (chosen | longest_selection).sum(dim=1).float().clamp_min(1.0)
            row = {
                "retained_target_rate": float(kept.float().mean()),
                "provisional_retained": float(survived_write.float().mean()),
                "top1": float(correct.float().mean()),
                "margin_over_longest_k": float(kept.float().mean()) - longest,
                "selection_iou_with_longest_k": float((intersection / union).mean()),
                "cosine_consolidation_ceiling": float(cosine_kept.float().mean()),
            }
            # knowledge-update questions invert the objective: their later
            # evidence supersedes the earlier session, so retaining the write
            # target is not obviously the right behaviour there.
            for index, label in enumerate(type_names):
                subset = payload["question_type"] == index
                if bool(subset.any()):
                    row[f"retained_{label}"] = float(kept[subset].float().mean())
                    row[f"top1_{label}"] = float(correct[subset].float().mean())
            rows.append(row)
        results[name] = {
            metric: {
                "mean": mean(row[metric] for row in rows),
                "std": pstdev(row[metric] for row in rows),
            }
            for metric in rows[0]
        }

    output = {
        "config": vars(args),
        "episodes": episodes,
        "traces": traces,
        "kept_slots": keep,
        "references": {
            "random_capacity": keep / traces,
            "longest_k_retention": longest,
            "longest_k_retention_by_type": per_type_longest,
            "query_visible_cosine_retention": query_visible,
            "event_visible_cosine_retention": event_visible,
            "episodes_by_type": {
                label: int((payload["question_type"] == index).sum())
                for index, label in enumerate(type_names)
                if bool((payload["question_type"] == index).any())
            },
        },
        "variants": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
