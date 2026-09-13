from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import torch
from torch import Tensor

from neural_memory.generator_probe import build_memory_prompt, score_answer_nll
from neural_memory.global_feedback import load_global_feedback_banks
from neural_memory.longmemeval import iter_longmemeval_revisits
from train_revisit_write_recall import add_causal_write_features, train_model


@torch.no_grad()
def selected_mask(
    model: torch.nn.Module,
    candidates: Tensor,
    feedback: Tensor,
    *,
    causal_features: bool,
    causal_feature_mode: str,
    device: torch.device,
) -> Tensor:
    """Return the writer's hard capacity decision for every evaluation episode."""
    if causal_features:
        candidates, feedback = add_causal_write_features(
            candidates, feedback, feature_mode=causal_feature_mode
        )
    candidates, feedback = (tensor.to(device) for tensor in (candidates, feedback))
    masks = torch.ones(candidates.shape[:2], dtype=torch.bool, device=device)
    model.eval()
    state = model.write(candidates, torch.zeros_like(feedback), masks)
    return state.selected.cpu()


def length_matched_selection(
    slot_lengths: list[list[int]], learned_rows: list[list[int]], keep: int
) -> list[list[int]]:
    """Pick a different slot set whose total length is closest to the learned one.

    The learned writer prefers longer sessions, and a longer prompt raises answer
    perplexity on its own, so a length-matched control separates what the writer
    selected from how much text it selected.
    """
    matched: list[list[int]] = []
    for lengths, learned in zip(slot_lengths, learned_rows, strict=True):
        target_length = sum(lengths[slot] for slot in learned)
        options = [
            list(combination)
            for combination in combinations(range(len(lengths)), keep)
            if list(combination) != sorted(learned)
        ]
        if not options:
            raise ValueError("no alternative slot set exists at this capacity")
        matched.append(
            min(options, key=lambda option: abs(sum(lengths[slot] for slot in option) - target_length))
        )
    return matched


def condition_indices(
    learned: Tensor, targets: Tensor, *, traces: int, keep: int, seed: int
) -> dict[str, list[list[int]]]:
    """Build the memory slot selection of every compared condition.

    Every capacity-limited condition holds exactly ``keep`` slots, including the
    oracle, so prompt length cannot explain a likelihood difference between them.
    """
    generator = torch.Generator().manual_seed(seed)
    episodes = len(targets)
    random_rows = [
        torch.randperm(traces, generator=generator)[:keep].sort().values.tolist()
        for _ in range(episodes)
    ]
    oracle_rows: list[list[int]] = []
    for index, target in enumerate(targets):
        others = [slot for slot in random_rows[index] if slot != int(target)]
        oracle_rows.append(sorted([int(target), *others[: keep - 1]]))
    return {
        "no_memory": [[] for _ in range(episodes)],
        "learned": [row.nonzero().flatten().tolist() for row in learned],
        "recency": [list(range(traces - keep, traces)) for _ in range(episodes)],
        "random": random_rows,
        "oracle": oracle_rows,
        "all_traces": [list(range(traces)) for _ in range(episodes)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Does the written memory make the gold answer more likely?"
    )
    parser.add_argument("--features", type=Path, default=Path("artifacts/longmemeval_bge.pt"))
    parser.add_argument("--input", type=Path, default=Path("artifacts/longmemeval_s_cleaned.json"))
    parser.add_argument("--generator", default="google/gemma-3-270m")
    parser.add_argument("--generator-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--write-steps", type=int, default=1000)
    parser.add_argument("--recall-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--keep-ratio", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--causal-features", action="store_true")
    # LongMemEval-S sessions hold about 11,000 characters and the evidence
    # sentence is rarely in the opening lines, so a short cap silently deletes
    # the very signal the oracle condition is supposed to supply.
    parser.add_argument("--max-session-chars", type=int, default=20000)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--score-batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 scores every eval episode")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
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
        shuffle_write_teacher=False,
        causal_features=args.causal_features,
        device=device,
    )
    learned = selected_mask(
        model,
        banks.eval_candidates,
        banks.eval_feedback,
        causal_features=args.causal_features,
        causal_feature_mode="all",
        device=device,
    )

    traces = banks.eval_candidates.shape[1]
    examples = [
        example
        for example in iter_longmemeval_revisits(args.input, candidates=traces)
        if example.split == "eval"
    ]
    if len(examples) != len(banks.eval_targets):
        raise RuntimeError(
            f"raw eval episodes ({len(examples)}) do not align with the feature artifact "
            f"({len(banks.eval_targets)}); regenerate the features from this JSON"
        )
    scorable = [
        index for index, example in enumerate(examples) if example.answer.strip()
    ]
    if not scorable:
        raise RuntimeError("no evaluation episode carries a gold answer")
    if args.limit:
        scorable = scorable[: args.limit]

    keep = int(learned[0].sum())
    conditions = condition_indices(
        learned, banks.eval_targets, traces=traces, keep=keep, seed=args.seed
    )
    slot_lengths = [
        [min(len(session), args.max_session_chars) for session in example.candidates]
        for example in examples
    ]
    conditions["length_matched"] = length_matched_selection(
        slot_lengths, conditions["learned"], keep
    )
    answers = [examples[index].answer for index in scorable]
    results: dict[str, dict[str, float]] = {}
    per_episode: dict[str, list[float]] = {}
    for name, selections in conditions.items():
        prompts = [
            build_memory_prompt(
                [examples[index].candidates[slot] for slot in selections[index]],
                examples[index].question,
                max_session_chars=args.max_session_chars,
            )
            for index in scorable
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
        lengths = torch.tensor([float(len(prompt)) for prompt in prompts])
        results[name] = {
            "answer_nll": float(scores.mean()),
            "answer_nll_std": float(scores.std(unbiased=False)),
            "mean_prompt_chars": float(lengths.mean()),
            "truncated_prompt_rate": truncated / len(prompts),
        }

    baseline = torch.tensor(per_episode["no_memory"])
    for name, scores in per_episode.items():
        values = torch.tensor(scores)
        results[name]["gain_over_no_memory"] = float((baseline - values).mean())
        results[name]["win_rate_over_no_memory"] = float((values < baseline).float().mean())
    learned_scores = torch.tensor(per_episode["learned"])
    for name in ("random", "recency", "oracle", "length_matched"):
        values = torch.tensor(per_episode[name])
        results["learned"][f"win_rate_over_{name}"] = float(
            (learned_scores < values).float().mean()
        )
    for name, selections in conditions.items():
        hits = torch.tensor(
            [float(int(banks.eval_targets[index]) in selections[index]) for index in scorable]
        )
        results[name]["evidence_hit_rate"] = float(hits.mean())
    # Does the generator benefit specifically when the writer kept the evidence?
    kept = torch.tensor(
        [bool(learned[index, banks.eval_targets[index]]) for index in scorable]
    )
    for label, mask in (("with_evidence", kept), ("without_evidence", ~kept)):
        if bool(mask.any()):
            results["learned"][f"answer_nll_{label}"] = float(learned_scores[mask].mean())
            results["learned"][f"episodes_{label}"] = int(mask.sum())
            # Same-subset reference: episodes the writer solved may simply be
            # easier, so the split only means something against this baseline.
            results["learned"][f"no_memory_nll_{label}"] = float(baseline[mask].mean())
            results["learned"][f"oracle_nll_{label}"] = float(
                torch.tensor(per_episode["oracle"])[mask].mean()
            )

    retained = torch.stack(
        [learned[index, banks.eval_targets[index]] for index in scorable]
    )
    # The endpoint only carries information about the writer once the generator
    # demonstrably benefits from the human-annotated evidence session itself.
    validity = {
        "oracle_prompts_intact": results["oracle"]["truncated_prompt_rate"] == 0.0,
        "oracle_beats_no_memory": results["oracle"]["answer_nll"]
        < results["no_memory"]["answer_nll"],
        "oracle_beats_random": results["oracle"]["answer_nll"]
        < results["random"]["answer_nll"],
    }
    output = {
        "validity": validity,
        "instrument_usable": all(validity.values()),
        "config": vars(args),
        "generator": args.generator,
        "scored_episodes": len(scorable),
        "traces": traces,
        "kept_slots": keep,
        "evidence_retained_rate": float(retained.float().mean()),
        "conditions": results,
        "per_episode": per_episode,
        "learned_kept_evidence": kept.tolist(),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
