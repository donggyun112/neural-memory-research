from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neural_memory.context_memory import MultiTraceRecallMemory
from neural_memory.contextbench_data import ContextEpisode, load_contextbench_episodes
from neural_memory.text_encoders import pool_mean
from train_context_activation_cv import fit, measure, shuffle_within_projects


def project_folds(episodes: list[ContextEpisode], folds: int) -> list[list[str]]:
    counts: dict[str, int] = {}
    for episode in episodes:
        counts[episode.project] = counts.get(episode.project, 0) + 1
    if folds < 2 or folds > len(counts):
        raise ValueError("folds must be between 2 and the number of projects")
    groups = [[] for _ in range(folds)]
    sizes = [0] * folds
    for project, size in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(folds), key=lambda index: sizes[index])
        groups[fold].append(project)
        sizes[fold] += size
    return groups


def encode_batch(
    backbone: nn.Module,
    tokenizer: object,
    texts: list[str],
    *,
    device: torch.device,
    max_length: int,
) -> Tensor:
    encoded = tokenizer(  # type: ignore[operator]
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    hidden = backbone(**encoded).last_hidden_state
    return F.normalize(pool_mean(hidden, encoded["attention_mask"]).float(), dim=-1)


def load_backbone(
    model_name: str,
    *,
    device: torch.device,
    lora_rank: int,
    lora_layers: int,
) -> tuple[nn.Module, object]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    backbone = AutoModel.from_pretrained(
        model_name, dtype=torch.float32, local_files_only=True
    ).to(device)
    if lora_rank > 0:
        from peft import LoraConfig, get_peft_model

        total_layers = backbone.config.num_hidden_layers
        if lora_layers < 1 or lora_layers > total_layers:
            raise ValueError("lora_layers must be between 1 and the model layer count")
        first_layer = max(0, total_layers - lora_layers)
        target_modules = [
            f"layers.{layer}.self_attn.{projection}"
            for layer in range(first_layer, total_layers)
            for projection in ("q_proj", "v_proj")
        ]
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
        )
        backbone = get_peft_model(backbone, config)
        backbone.train()
    else:
        backbone.requires_grad_(False)
        backbone.eval()
    return backbone, tokenizer


def sample_pairs(
    episodes: list[ContextEpisode], rng: random.Random, batch_size: int
) -> tuple[list[str], list[str], Tensor]:
    selected = [rng.choice(episodes) for _ in range(batch_size)]
    candidates = [rng.choice(episode.candidates) for episode in selected]
    return (
        [episode.query_context for episode in selected],
        [candidate.context for candidate in candidates],
        torch.tensor([candidate.active for candidate in candidates], dtype=torch.float32),
    )


def sample_episode_sets(
    episodes: list[ContextEpisode], rng: random.Random, batch_size: int
) -> tuple[list[str], list[list[str]], Tensor, Tensor]:
    selected = [rng.choice(episodes) for _ in range(batch_size)]
    max_candidates = max(len(episode.candidates) for episode in selected)
    targets = torch.zeros(batch_size, max_candidates, dtype=torch.float32)
    masks = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    candidate_groups: list[list[str]] = []
    for row, episode in enumerate(selected):
        group = [candidate.context for candidate in episode.candidates]
        candidate_groups.append(group)
        for column, candidate in enumerate(episode.candidates):
            targets[row, column] = candidate.active
            masks[row, column] = True
    return (
        [episode.query_context for episode in selected],
        candidate_groups,
        targets,
        masks,
    )


def encode_unique(
    backbone: nn.Module,
    tokenizer: object,
    episodes: list[ContextEpisode],
    *,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> dict[str, Tensor]:
    texts = list(
        dict.fromkeys(
            text
            for episode in episodes
            for text in (
                episode.query_context,
                *(candidate.context for candidate in episode.candidates),
            )
        )
    )
    chunks: list[Tensor] = []
    backbone.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunks.append(
                encode_batch(
                    backbone,
                    tokenizer,
                    texts[start : start + batch_size],
                    device=device,
                    max_length=max_length,
                ).cpu()
            )
    return dict(zip(texts, torch.cat(chunks), strict=True))


def evaluate(
    memory: MultiTraceRecallMemory,
    cache: dict[str, Tensor],
    episodes: list[ContextEpisode],
    *,
    device: torch.device,
) -> dict[str, object]:
    max_candidates = max(len(episode.candidates) for episode in episodes)
    dimension = next(iter(cache.values())).shape[-1]
    candidates = torch.zeros(len(episodes), max_candidates, dimension)
    queries = torch.zeros(len(episodes), dimension)
    targets = torch.zeros(len(episodes), max_candidates, dtype=torch.bool)
    masks = torch.zeros(len(episodes), max_candidates, dtype=torch.bool)
    project_names = sorted({episode.project for episode in episodes})
    project_index = {name: index for index, name in enumerate(project_names)}
    projects = torch.zeros(len(episodes), dtype=torch.long)
    for row, episode in enumerate(episodes):
        queries[row] = cache[episode.query_context]
        projects[row] = project_index[episode.project]
        for column, candidate in enumerate(episode.candidates):
            candidates[row, column] = cache[candidate.context]
            targets[row, column] = candidate.active
            masks[row, column] = True
    memory.eval()
    with torch.inference_mode():
        logits = memory(candidates.to(device), queries.to(device)).cpu()
        shuffled = shuffle_within_projects(queries, projects)
        shuffled_logits = memory(candidates.to(device), shuffled.to(device)).cpu()
    return {
        "joint": asdict(measure(logits, targets, masks)),
        "joint_shuffled_query": asdict(measure(shuffled_logits, targets, masks)),
    }


def initialize_memory(
    feature_path: Path,
    episodes: list[ContextEpisode],
    heldout: set[str],
    *,
    steps: int,
    batch_size: int,
    memory_dim: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> MultiTraceRecallMemory:
    payload = torch.load(feature_path, map_location="cpu", weights_only=True)
    candidates = payload["all_candidates"]
    queries = payload["all_queries"]
    targets = payload["all_targets"]
    masks = payload["all_masks"]
    if len(episodes) != len(targets):
        raise ValueError("initial feature artifact does not match the episode count")
    expected = torch.zeros_like(targets)
    expected_mask = torch.zeros_like(masks)
    for row, episode in enumerate(episodes):
        for column, candidate in enumerate(episode.candidates):
            expected[row, column] = candidate.active
            expected_mask[row, column] = True
    if not torch.equal(targets, expected) or not torch.equal(masks, expected_mask):
        raise ValueError("initial feature artifact labels do not match the episodes")
    train_mask = torch.tensor(
        [episode.project not in heldout for episode in episodes], dtype=torch.bool
    )
    model = fit(
        candidates[train_mask],
        queries[train_mask],
        targets[train_mask],
        masks[train_mask],
        mode="joint",
        steps=steps,
        batch_size=batch_size,
        memory_dim=memory_dim,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
        architecture="relation",
        alignment_weight=1.0,
    )
    if not isinstance(model, MultiTraceRecallMemory):
        raise TypeError("relation pretraining returned an unexpected model")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="One-fold Gemma LoRA recall pilot")
    parser.add_argument("--input", type=Path, default=Path("local-data/contextbench.jsonl"))
    parser.add_argument("--model", default="google/gemma-3-270m")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--heldout-fold", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--candidate-sampling", choices=("pair", "episode"), default="pair"
    )
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--max-eval-episodes",
        type=int,
        default=0,
        help="Deterministic held-out subset for a fast pilot; zero evaluates all",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument(
        "--init-features", type=Path, default=Path("artifacts/contextbench-gemma-mean.pt")
    )
    parser.add_argument("--pretrain-steps", type=int, default=2000)
    parser.add_argument("--pretrain-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--memory-finetune-learning-rate", type=float, default=0.0)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    episodes = load_contextbench_episodes(args.input)
    folds = project_folds(episodes, args.folds)
    if args.heldout_fold < 0 or args.heldout_fold >= len(folds):
        raise ValueError("heldout fold is out of range")
    heldout = set(folds[args.heldout_fold])
    train_episodes = [episode for episode in episodes if episode.project not in heldout]
    eval_episodes = [episode for episode in episodes if episode.project in heldout]
    if args.max_eval_episodes > 0:
        eval_episodes = eval_episodes[: args.max_eval_episodes]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    backbone, tokenizer = load_backbone(
        args.model,
        device=device,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
    )
    if args.pretrain_steps > 0:
        memory = initialize_memory(
            args.init_features,
            episodes,
            heldout,
            steps=args.pretrain_steps,
            batch_size=args.pretrain_batch_size,
            memory_dim=args.memory_dim,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=device,
        )
    else:
        memory = MultiTraceRecallMemory(backbone.config.hidden_size, args.memory_dim).to(device)
    memory_parameters = list(memory.parameters())
    lora_parameters = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    parameter_groups = [
        {"params": memory_parameters, "lr": args.memory_finetune_learning_rate}
    ]
    if lora_parameters:
        parameter_groups.append(
            {"params": lora_parameters, "lr": args.lora_learning_rate}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-3)
    memory.train()
    losses: list[float] = []
    for _ in range(args.steps):
        if args.candidate_sampling == "episode":
            query_texts, candidate_groups, targets, training_masks = sample_episode_sets(
                train_episodes, rng, args.batch_size
            )
            candidate_texts = [text for group in candidate_groups for text in group]
        else:
            query_texts, candidate_texts, pair_targets = sample_pairs(
                train_episodes, rng, args.batch_size
            )
            targets = pair_targets[:, None]
            training_masks = torch.ones_like(targets, dtype=torch.bool)
        if lora_parameters:
            features = encode_batch(
                backbone,
                tokenizer,
                [*query_texts, *candidate_texts],
                device=device,
                max_length=args.max_length,
            )
        else:
            with torch.no_grad():
                features = encode_batch(
                    backbone,
                    tokenizer,
                    [*query_texts, *candidate_texts],
                    device=device,
                    max_length=args.max_length,
                )
        queries = features[: len(query_texts)]
        flat_candidates = features[len(query_texts) :]
        if args.candidate_sampling == "episode":
            candidates = torch.zeros(
                len(query_texts),
                targets.shape[1],
                flat_candidates.shape[-1],
                device=device,
            )
            offset = 0
            for row, group in enumerate(candidate_groups):
                candidates[row, : len(group)] = flat_candidates[offset : offset + len(group)]
                offset += len(group)
        else:
            candidates = flat_candidates[:, None]
        logits = memory(candidates, queries)
        loss = F.binary_cross_entropy_with_logits(
            logits[training_masks.to(device)], targets.to(device)[training_masks.to(device)]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    cache = encode_unique(
        backbone,
        tokenizer,
        eval_episodes,
        device=device,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
    )
    output = {
        "config": vars(args),
        "train_episodes": len(train_episodes),
        "eval_episodes": len(eval_episodes),
        "heldout_projects": len(heldout),
        "memory_parameters": sum(parameter.numel() for parameter in memory_parameters),
        "lora_parameters": sum(parameter.numel() for parameter in lora_parameters),
        "train_loss_first_10": (
            sum(losses[:10]) / min(10, len(losses)) if losses else None
        ),
        "train_loss_last_10": (
            sum(losses[-10:]) / min(10, len(losses)) if losses else None
        ),
        "metrics": evaluate(memory, cache, eval_episodes, device=device),
    }
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
