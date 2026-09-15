"""Does injecting memory change what the model would do, for the better?

Every measurement in this project ranked things. The requirement was a memory
that improves an agent's behaviour, and behaviour was never touched: the logs
record an agent that had no memory, so nothing could have been surfaced and no
observational number can show surfacing helping.

This intervenes. Open-SWE trajectories are all `resolved`, so the action the
real agent took next is a known-good one. A frozen model is asked how likely
that action is given the trajectory so far, once with memory injected and once
without. If injection raises the likelihood of the good action, the memory
changed what the model would do, in the right direction.

Conditions all inject the same number of memory items so length cannot stand in
for content:

  no_memory       the recent context alone
  recency         the k actions immediately before the window
  similarity      the k earlier actions most like the current state
  oracle          the k earlier actions that a reader with the future in hand
                  would pick. The gate: if this does not beat no_memory, the
                  instrument cannot detect a good memory and nothing else in the
                  table is readable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from neural_memory.generator_probe import score_answer_nll


def build_prompt(recent: list[str], memory: list[str], question: str) -> str:
    blocks = []
    if memory:
        blocks.append("Earlier steps that may be relevant:\n" + "\n".join(memory))
    blocks.append("Recent steps:\n" + "\n".join(recent))
    blocks.append(f"{question}\nNext step:")
    return "\n\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("local-data/open-swe-v1.jsonl"))
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/open_swe_tool_stream.pt")
    )
    parser.add_argument("--generator", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--generator-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--trajectories", type=int, default=60)
    parser.add_argument("--min-calls", type=int, default=64, help="must match the artifact")
    parser.add_argument("--limit", type=int, default=1200, help="must match the artifact")
    parser.add_argument("--recent", type=int, default=6, help="steps of immediate context")
    parser.add_argument("--keep", type=int, default=4, help="memory items per condition")
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # The same rendering the stream artifact used, so embeddings and text line up.
    from prepare_claude_tool_stream import open_swe_streams

    # These have to be the values the feature artifact was built with, or the
    # text and the embeddings describe different streams and the alignment check
    # below silently discards nearly everything.
    streams, _ = open_swe_streams(args.traces, minimum=args.min_calls, limit=args.limit)
    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    turns = F.normalize(payload["turns"], dim=-1)
    offsets = payload["offsets"]
    usable = min(len(streams), len(offsets) - 1, args.trajectories)

    generator = np.random.default_rng(args.seed)
    names = ["no_memory", "recency", "similarity", "summary", "oracle"]
    prompts: dict[str, list[str]] = {name: [] for name in names}
    answers: list[str] = []
    for index in range(usable):
        stream = streams[index]
        start, stop = int(offsets[index]), int(offsets[index + 1])
        if stop - start != len(stream):
            continue
        position = int(generator.integers(args.warmup, len(stream) - 1))
        recent = stream[position - args.recent : position]
        earlier = list(range(0, position - args.recent))
        if len(earlier) < args.keep:
            continue
        cue = turns[start + position - 1]
        similarity = (turns[start : start + len(earlier)] @ cue).numpy()
        # The oracle looks at the action being predicted, which no reader can do.
        future = turns[start + position]
        ahead = (turns[start : start + len(earlier)] @ future).numpy()
        # The fly has no selection step. About two thousand Kenyon cells converge
        # on thirty-four output neurons whose ensemble represents the answer
        # combinatorially, and long-term retrieval needs all of them, each
        # carrying part of the context. Nothing decides which memory is relevant.
        # The equivalent here is to compress the whole memory into `keep`
        # channels and show one representative of each, rather than to pick the
        # `keep` items that look relevant -- which is exactly the decision Phase
        # 71 found undetermined.
        block = turns[start : start + len(earlier)].numpy()
        centres = block[np.linspace(0, len(block) - 1, args.keep).astype(int)]
        for _ in range(8):
            assignment = np.argmax(centres @ block.T, axis=0)
            for channel in range(args.keep):
                members = block[assignment == channel]
                if len(members):
                    centres[channel] = members.mean(axis=0)
            norms = np.linalg.norm(centres, axis=1, keepdims=True)
            centres = np.divide(centres, norms, out=centres, where=norms > 0)
        summary = sorted({int(np.argmax(block @ centre)) for centre in centres})
        picks = {
            "no_memory": [],
            "recency": earlier[-args.keep :],
            "similarity": sorted(np.argsort(-similarity)[: args.keep].tolist()),
            "summary": summary,
            "oracle": sorted(np.argsort(-ahead)[: args.keep].tolist()),
        }
        for name in names:
            prompts[name].append(
                build_prompt(
                    recent,
                    [stream[slot] for slot in picks[name]],
                    "What is the next step this agent takes?",
                )
            )
        answers.append(stream[position])

    if not answers:
        raise RuntimeError("no trajectory produced a usable position")

    results: dict[str, dict[str, float]] = {}
    per_item: dict[str, np.ndarray] = {}
    for name in names:
        scores, truncated = score_answer_nll(
            prompts[name],
            answers,
            model_name=args.generator,
            device=args.device,
            batch_size=1,
            max_length=args.max_length,
            dtype=args.generator_dtype,
        )
        per_item[name] = scores.numpy()
        results[name] = {
            "action_nll": float(scores.mean()),
            "truncated_rate": truncated / len(prompts[name]),
            "mean_prompt_chars": float(np.mean([len(text) for text in prompts[name]])),
        }

    baseline = per_item["no_memory"]
    print(f"{len(answers)} trajectories, {args.keep} memory items per condition\n")
    print(f"{'condition':>12} {'action NLL':>11} {'gain':>9} {'win rate':>9} {'chars':>8}")
    for name in names:
        gap = baseline - per_item[name]
        results[name]["gain_over_no_memory"] = float(gap.mean())
        results[name]["win_rate"] = float((gap > 0).mean())
        print(
            f"{name:>12} {results[name]['action_nll']:11.4f}"
            f" {results[name]['gain_over_no_memory']:+9.4f}"
            f" {results[name]['win_rate']:9.3f} {results[name]['mean_prompt_chars']:8.0f}"
        )

    # Answer likelihood is heavy-tailed and Phase 33 was retracted for reading a
    # mean over it. Recency leads on the mean here while similarity leads on the
    # win rate, which is that same disagreement, so the differences are paired.
    print(f"\n{'comparison':>28} {'difference':>11} {'95% interval':>22}")
    head_to_head = {}
    for left, right in (
        ("summary", "similarity"),
        ("summary", "recency"),
        ("summary", "no_memory"),
        ("similarity", "recency"),
        ("oracle", "summary"),
    ):
        gap = per_item[right] - per_item[left]
        maker = np.random.default_rng(7)
        draws = [gap[maker.integers(0, len(gap), len(gap))].mean() for _ in range(5000)]
        low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        head_to_head[f"{left} minus {right}"] = {
            "difference": float(gap.mean()),
            "low": low,
            "high": high,
            "win_rate": float((gap > 0).mean()),
        }
        print(
            f"{left + ' over ' + right:>28} {gap.mean():+11.4f}  [{low:+.4f}, {high:+.4f}]"
            f" {'resolved' if low > 0 or high < 0 else 'NOT resolved'}"
        )

    usable_instrument = results["oracle"]["gain_over_no_memory"] > 0
    print(f"\ninstrument usable (oracle beats no memory): {usable_instrument}")
    if not usable_instrument:
        print("  -> nothing above is readable; see Phase 33")
    output = {
        "config": vars(args),
        "trajectories": len(answers),
        "instrument_usable": bool(usable_instrument),
        "conditions": results,
        "head_to_head": head_to_head,
        "per_item": {name: values.tolist() for name, values in per_item.items()},
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
