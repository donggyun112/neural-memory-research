from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from eval_question_retrieval import (
    cosine_scores,
    load_episodes,
    memory_scores,
    split_episodes,
    with_distractors,
)
from neural_memory.generator_probe import build_memory_prompt, score_answer_nll
from neural_memory.longmemeval import iter_longmemeval_deferred, iter_longmemeval_revisits
from neural_memory.trainable import TrainableMemory


def top_slots(scores: Tensor, keep: int) -> list[list[int]]:
    return [sorted(row.topk(keep).indices.tolist()) for row in scores]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Does the store's own choice of sessions make the gold answer more likely?"
    )
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_deferred.pt")
    )
    parser.add_argument("--input", type=Path, default=Path("artifacts/longmemeval_s_cleaned.json"))
    parser.add_argument("--generator", default="google/gemma-3-270m")
    parser.add_argument("--generator-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--key-dim", type=int, default=512)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--fixed-active", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-session-chars", type=int, default=20000)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--score-batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 scores every held-out episode")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    candidates, queries, targets, train_count = load_episodes(args.features)
    episodes, traces, features = candidates.shape

    if train_count is None:
        examples = list(iter_longmemeval_deferred(args.input, candidates=traces))
    else:
        # The revisit artifact groups its rows by split, so the raw episodes have
        # to be regrouped the same way before they line up with the features.
        raw = list(iter_longmemeval_revisits(args.input, candidates=traces))
        examples = [row for row in raw if row.split == "train"]
        examples += [row for row in raw if row.split == "eval"]
        if len(examples) - len(
            [row for row in raw if row.split == "train"]
        ) != episodes - train_count:
            raise RuntimeError("the artifact's split sizes do not match the raw episodes")
    if len(examples) != episodes:
        raise RuntimeError(
            f"raw episodes ({len(examples)}) do not align with the feature artifact "
            f"({episodes}); regenerate the features from this JSON"
        )

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_index, eval_index = split_episodes(episodes, train_count, args.holdout, generator)
    loaded = with_distractors(candidates, 0, generator)

    model = TrainableMemory(
        features,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        adaptive=False,
        fixed_active=args.fixed_active,
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    model.train()
    for step in range(args.steps):
        pick = int(train_index[step % len(train_index)])
        logits = model.probe(loaded[pick], queries[pick].unsqueeze(0))
        loss = F.cross_entropy(logits, targets[pick].unsqueeze(0))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    held = eval_index[: args.limit] if args.limit else eval_index
    held_candidates, held_queries = loaded[held], queries[held]
    held_targets = targets[held]

    memory = memory_scores(model, held_candidates, held_queries, 1.0)
    cosine = cosine_scores(held_candidates, held_queries)
    picks = torch.randperm(traces, generator=generator)[: args.keep].sort().values.tolist()
    conditions = {
        "no_memory": [[] for _ in held],
        "memory": top_slots(memory, args.keep),
        "cosine": top_slots(cosine, args.keep),
        "random": [picks for _ in held],
        "recency": [list(range(traces - args.keep, traces)) for _ in held],
        # Every condition holds the same slot count, so prompt length cannot
        # stand in for the choice of session.
        "oracle": [
            sorted({int(target), *[slot for slot in picks if slot != int(target)][: args.keep - 1]})
            for target in held_targets
        ],
        # A deferred episode's answer is spread over two evidence sessions and
        # only the first is a candidate, so the oracle above may be handing the
        # generator half a fact. These two say whether that is what breaks it.
        "evidence_only": [[int(target)] for target in held_targets],
    }
    sessions = {
        name: [
            [examples[int(index)].candidates[slot] for slot in slots]
            for index, slots in zip(held, selections, strict=True)
        ]
        for name, selections in conditions.items()
    }
    if train_count is None:
        # Only the deferred episodes hold a second evidence session outside the
        # candidate pool; a revisit episode's evidence is complete in one slot.
        sessions["evidence_and_consolidation"] = [
            [examples[int(index)].candidates[int(target)], examples[int(index)].consolidation]
            for index, target in zip(held, held_targets, strict=True)
        ]
        conditions["evidence_and_consolidation"] = [[int(target)] for target in held_targets]

    answers = [examples[int(index)].answer for index in held]
    if not all(answer.strip() for answer in answers):
        raise RuntimeError("an evaluation episode carries no gold answer")

    results: dict[str, dict[str, float]] = {}
    per_episode: dict[str, list[float]] = {}
    started = time.monotonic()
    for order, (name, selections) in enumerate(conditions.items(), start=1):
        print(
            f"[{time.monotonic() - started:7.1f}s] scoring {order}/{len(conditions)}: {name}",
            file=sys.stderr,
            flush=True,
        )
        prompts = [
            build_memory_prompt(
                texts,
                examples[int(index)].question,
                max_session_chars=args.max_session_chars,
            )
            for index, texts in zip(held, sessions[name], strict=True)
        ]
        scores, truncated = score_answer_nll(
            prompts,
            answers,
            model_name=args.generator,
            device=args.device,
            batch_size=args.score_batch_size,
            max_length=args.max_length,
            dtype=args.generator_dtype,
        )
        per_episode[name] = scores.tolist()
        results[name] = {
            "answer_nll": float(scores.mean()),
            "answer_nll_std": float(scores.std(unbiased=False)),
            "mean_prompt_chars": float(
                torch.tensor([float(len(prompt)) for prompt in prompts]).mean()
            ),
            "truncated_prompt_rate": truncated / len(prompts),
            "evidence_hit_rate": float(
                torch.tensor(
                    [
                        float(int(target) in slots)
                        for target, slots in zip(held_targets, selections, strict=True)
                    ]
                ).mean()
            ),
        }

    baseline = torch.tensor(per_episode["no_memory"])
    for name, scores in per_episode.items():
        values = torch.tensor(scores)
        results[name]["gain_over_no_memory"] = float((baseline - values).mean())
        results[name]["win_rate_over_no_memory"] = float((values < baseline).float().mean())
    memory_nll = torch.tensor(per_episode["memory"])
    for name in ("cosine", "random", "recency", "oracle"):
        results["memory"][f"win_rate_over_{name}"] = float(
            (memory_nll < torch.tensor(per_episode[name])).float().mean()
        )

    # The endpoint only says anything about a reader once the generator can be
    # shown to benefit from evidence handed to it directly. The weakest form of
    # that is the whole annotated evidence set, alone, against an empty prompt.
    best_evidence = min(
        results[name]["answer_nll"]
        for name in ("oracle", "evidence_only", "evidence_and_consolidation")
        if name in results
    )
    validity = {
        "oracle_prompts_intact": results["oracle"]["truncated_prompt_rate"] == 0.0,
        "oracle_beats_no_memory": results["oracle"]["answer_nll"]
        < results["no_memory"]["answer_nll"],
        "oracle_beats_random": results["oracle"]["answer_nll"] < results["random"]["answer_nll"],
        "any_evidence_beats_no_memory": best_evidence < results["no_memory"]["answer_nll"],
    }
    output = {
        "validity": validity,
        "instrument_usable": all(validity.values()),
        "config": vars(args),
        "scored_episodes": len(held),
        "traces": traces,
        "conditions": results,
        "per_episode": per_episode,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
