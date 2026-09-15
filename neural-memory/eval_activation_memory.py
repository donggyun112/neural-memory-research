"""Memory as a bias on the computation, not as text in the prompt.

Phase 75 ruled out the representation hypothesis and left a sharper one: the
better a space captures what is happening now, the more the nearest stored item
duplicates the context already in the prompt. Similarity retrieval competes with
recency for the same tokens, and a better encoder makes that competition worse.

If that is the problem, injecting memory as tokens is the wrong channel
regardless of what is chosen. The fly does not do it either — Kenyon cells read
the whole weight matrix and the output biases behaviour, with nothing selected
and nothing inserted into the input.

So this adds a memory vector to the residual stream at one layer and leaves the
prompt alone. The vector is the mean of the stored actions' own hidden states,
which is the spread summary taken to its limit: every item contributes, nothing
is chosen. Scoring is the same known-good action, the same paired test, the same
generator; the only change is the channel the memory arrives through.

No training. A forward hook and a scale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("local-data/open-swe-mixed.jsonl"))
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/open_swe_qwen_stream.pt")
    )
    parser.add_argument("--generator", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--trajectories", type=int, default=300)
    parser.add_argument("--recent", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--min-calls", type=int, default=64)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--layer", type=int, default=14, help="block to bias")
    parser.add_argument(
        "--scales",
        default="0.05,0.1,0.2,0.4",
        help="bias size as a fraction of the residual stream's own norm",
    )
    parser.add_argument(
        "--centre",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="remove the shared direction from the action vectors",
    )
    parser.add_argument("--expansion", type=int, default=2048, help="Kenyon cells")
    parser.add_argument("--cells", type=int, default=64, help="cells left active per code")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from neural_memory.familiarity import sparse_code
    from prepare_claude_tool_stream import open_swe_streams
    from transformers import AutoModelForCausalLM, AutoTokenizer

    streams, _ = open_swe_streams(args.traces, args.min_calls, args.limit)
    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    turns = payload["turns"]
    offsets = payload["offsets"]
    if args.centre:
        # Mean-pooled hidden states sit in a narrow cone: raw actions average
        # 0.87 cosine to each other, so any summary of one history lands 0.98
        # from the summary of an unrelated one and own-versus-foreign has no
        # contrast to detect. Removing the shared direction takes that to 0.06.
        turns = F.normalize(turns - turns.mean(0), dim=1)
    projection = np.random.default_rng(args.seed).normal(
        size=(turns.shape[1], args.expansion)
    ) / np.sqrt(turns.shape[1])

    tokenizer = AutoTokenizer.from_pretrained(args.generator)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.generator, dtype=getattr(torch, args.dtype)
    ).to(args.device)
    model.eval()
    block = model.model.layers[args.layer]

    bias = {"vector": None, "scale": 0.0}

    def hook(module, inputs, output):
        if bias["vector"] is None or bias["scale"] == 0.0:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        # A unit vector is nothing next to a residual stream of norm ~50, so the
        # size is set as a fraction of what is already flowing through.
        step = bias["vector"].to(hidden.dtype).to(hidden.device)
        shifted = hidden + bias["scale"] * hidden.norm(dim=-1, keepdim=True) * step
        return (shifted, *output[1:]) if isinstance(output, tuple) else shifted

    handle = block.register_forward_hook(hook)

    generator = np.random.default_rng(args.seed)
    noise = np.random.default_rng(args.seed + 1)
    cases = []
    for index in range(min(len(streams), len(offsets) - 1, args.trajectories)):
        stream = streams[index]
        start, stop = int(offsets[index]), int(offsets[index + 1])
        if stop - start != len(stream) or len(stream) < args.warmup + args.recent + 2:
            continue
        position = int(generator.integers(args.warmup, len(stream) - 1))
        memory = turns[start : start + position - args.recent]
        if len(memory) < 4:
            continue
        prompt = "Recent steps:\n" + "\n".join(stream[position - args.recent : position])
        prompt += "\n\nWhat is the next step this agent takes?\nNext step:"
        # Every stored action contributes equally. This is the spread summary
        # with the channel count taken to the size of the memory.
        # The fly's readout is not the average of what was stored, it is the
        # whole weight matrix applied to the present state: every item
        # contributes in proportion to its overlap with now, and none is picked.
        cue = F.normalize(turns[start + position - args.recent : start + position].mean(0), dim=0)
        # Dense overlaps sit in a narrow positive band, so the weighted sum
        # collapses back to the flat mean. The fly avoids that by expanding into
        # sparse cells first, where codes are near-orthogonal; values stay dense.
        codes = sparse_code(memory.numpy(), projection, args.cells)
        weights = codes @ sparse_code(cue.numpy()[None], projection, args.cells)[0]
        sparse = torch.from_numpy(weights.astype("float32")) @ memory
        cases.append(
            {
                "prompt": prompt,
                "answer": stream[position],
                "index": index,
                "memory": F.normalize(memory.mean(0), dim=0),
                "associative": F.normalize((memory @ cue) @ memory, dim=0),
                "sparse": F.normalize(sparse, dim=0),
            }
        )

    if not cases:
        raise RuntimeError("no usable position")

    # Two controls, because a bias of any kind moves the NLL. `noise` says
    # whether direction matters at all; `foreign` says whether *this* history
    # matters or merely a plausible one.
    for offset, case in enumerate(cases):
        raw = torch.tensor(noise.normal(size=case["memory"].shape), dtype=torch.float32)
        case["noise"] = F.normalize(raw, dim=0)
        other = cases[(offset + len(cases) // 2) % len(cases)]
        case["foreign"] = other["memory"]
        case["foreign_associative"] = other["associative"]
        case["foreign_sparse"] = other["sparse"]

    scales = [float(value) for value in args.scales.split(",")]
    sources = (
        "memory",
        "associative",
        "sparse",
        "noise",
        "foreign",
        "foreign_associative",
        "foreign_sparse",
    )
    arms = [(source, scale) for source in sources for scale in scales]
    scores: dict[tuple[str, float], list[float]] = {arm: [] for arm in arms}
    baseline_scores: list[float] = []
    with torch.inference_mode():
        for case in cases:
            prompt_ids = tokenizer(case["prompt"], add_special_tokens=True)["input_ids"]
            answer = case["answer"].strip()
            answer_ids = tokenizer(f" {answer}", add_special_tokens=False)["input_ids"]
            answer_ids = answer_ids[: args.max_length // 2]
            budget = args.max_length - len(answer_ids)
            sequence = [*prompt_ids[-budget:], *answer_ids]
            tokens = torch.tensor([sequence], device=args.device)
            targets = torch.tensor([sequence[-len(answer_ids) :]], device=args.device)

            def score() -> float:
                logits = model(input_ids=tokens).logits[:, -len(answer_ids) - 1 : -1]
                return float(
                    F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1)
                    )
                )

            bias["vector"], bias["scale"] = None, 0.0
            baseline_scores.append(score())
            for source, scale in arms:
                bias["vector"], bias["scale"] = case[source], scale
                scores[(source, scale)].append(score())
    handle.remove()

    baseline = np.array(baseline_scores)

    def interval(gap: np.ndarray) -> tuple[float, float]:
        maker = np.random.default_rng(7)
        draws = [gap[maker.integers(0, len(gap), len(gap))].mean() for _ in range(3000)]
        return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    print(f"\n{len(cases)} positions, layer {args.layer}, bias added to the residual stream")
    print(f"no bias: {baseline.mean():.4f}\n")
    print(f"{'source':>9} {'scale':>7} {'action NLL':>11} {'gain':>9} {'95% interval':>22}")
    results: dict[str, dict] = {}
    for source, scale in arms:
        values = np.array(scores[(source, scale)])
        gap = baseline - values
        low, high = interval(gap)
        results[f"{source}@{scale}"] = {
            "action_nll": float(values.mean()),
            "gain": float(gap.mean()),
            "low": low,
            "high": high,
        }
        mark = " resolved" if low > 0 or high < 0 else " no"
        print(
            f"{source:>9} {scale:7.2f} {values.mean():11.4f} {gap.mean():+9.4f}"
            f"  [{low:+.4f}, {high:+.4f}]{mark}"
        )

    # The only comparison that says the memory carries this history rather than
    # any history is against the same read taken from another trajectory.
    print(f"\n{'comparison':>40} {'difference':>11} {'95% interval':>22}")
    pairs = [
        ("memory", "noise"),
        ("memory", "foreign"),
        ("associative", "noise"),
        ("associative", "foreign_associative"),
        ("sparse", "noise"),
        ("sparse", "foreign_sparse"),
        ("sparse", "memory"),
    ]
    for treatment, control in pairs:
        for scale in scales:
            gap = np.array(scores[(control, scale)]) - np.array(scores[(treatment, scale)])
            low, high = interval(gap)
            results[f"{treatment}_over_{control}@{scale}"] = {
                "difference": float(gap.mean()),
                "low": low,
                "high": high,
            }
            mark = " resolved" if low > 0 or high < 0 else " NOT resolved"
            label = f"{treatment} over {control} @ {scale}"
            print(f"{label:>40} {gap.mean():+11.4f}  [{low:+.4f}, {high:+.4f}]{mark}")

    output = {
        "config": vars(args),
        "positions": len(cases),
        "no_bias": float(baseline.mean()),
        "arms": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
