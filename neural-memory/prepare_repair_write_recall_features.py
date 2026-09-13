from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from neural_memory.codex_repair_chains import iter_codex_repair_episodes
from neural_memory.repair_chains import RepairEpisode, iter_repair_episodes
from neural_memory.text_encoders import encode_with_gemma


ACTION_LABELS = {"ignore": 0, "strengthen": 1, "revise": 2}


def unique_episode_texts(episodes: Sequence[RepairEpisode]) -> list[str]:
    return list(
        dict.fromkeys(
            text
            for episode in episodes
            for text in (
                *(candidate.context for candidate in episode.candidates),
                episode.outcome_context,
            )
        )
    )


def build_payload(
    episodes: Sequence[RepairEpisode], cache: dict[str, Tensor], feature_dim: int
) -> dict[str, Tensor | str]:
    max_candidates = max(len(episode.candidates) for episode in episodes)
    candidates = torch.zeros(len(episodes), max_candidates, feature_dim)
    recalls = torch.zeros(len(episodes), feature_dim)
    targets = torch.zeros(len(episodes), max_candidates, dtype=torch.bool)
    action_targets = torch.full((len(episodes), max_candidates), -100, dtype=torch.long)
    masks = torch.zeros(len(episodes), max_candidates, dtype=torch.bool)
    projects = torch.zeros(len(episodes), dtype=torch.long)
    for row, episode in enumerate(episodes):
        seed_text = f"{episode.project}:{episode.session_id}:{episode.family}"
        seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")
        shuffled = list(episode.candidates)
        random.Random(seed).shuffle(shuffled)
        for column, candidate in enumerate(shuffled):
            candidates[row, column] = cache[candidate.context]
            targets[row, column] = candidate.action != "ignore"
            action_targets[row, column] = ACTION_LABELS[candidate.action]
            masks[row, column] = True
        recalls[row] = cache[episode.outcome_context]
        projects[row] = (
            int.from_bytes(hashlib.sha256(episode.project.encode()).digest()[:8], "big")
            % (2**63 - 1)
        )
    return {
        "all_candidates": candidates,
        "all_write_queries": torch.zeros_like(recalls),
        "all_recall_queries": recalls,
        "all_targets": targets,
        "all_action_targets": action_targets,
        "all_masks": masks,
        "all_project_ids": projects,
        "write_context": "candidate-only",
        "recall_context": "later-failure-and-pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode real repair write/recall episodes")
    parser.add_argument("--root", type=Path, default=Path.home() / ".claude" / "projects")
    parser.add_argument(
        "--codex-root", type=Path, default=Path.home() / ".codex" / "sessions"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/repair-write-recall-gemma.pt")
    )
    parser.add_argument("--model", default="google/gemma-3-270m")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-per-action", type=int, default=4)
    parser.add_argument("--distractors", type=int, default=2)
    args = parser.parse_args()

    episodes = list(
        iter_repair_episodes(
            args.root,
            max_per_action=args.max_per_action,
            distractors=args.distractors,
        )
    )
    episodes.extend(
        iter_codex_repair_episodes(
            args.codex_root,
            max_per_action=args.max_per_action,
            distractors=args.distractors,
        )
    )
    if not episodes:
        raise RuntimeError("no local repair episodes found")
    texts = unique_episode_texts(episodes)
    encoded = encode_with_gemma(
        texts,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling="mean",
    )
    cache = dict(zip(texts, encoded, strict=True))
    payload = build_payload(episodes, cache, encoded.shape[-1])
    payload.update(
        {
            "encoder_model": args.model,
            "pooling": "mean",
            "max_length": args.max_length,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    labels = payload["all_action_targets"]
    masks = payload["all_masks"]
    assert isinstance(labels, Tensor) and isinstance(masks, Tensor)
    counts = {
        name: int((labels[masks] == index).sum()) for name, index in ACTION_LABELS.items()
    }
    print(
        f"saved episodes={len(episodes)},projects={payload['all_project_ids'].unique().numel()},"
        f"texts={len(texts)},labels={counts},dim={encoded.shape[-1]},raw_text=false "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
