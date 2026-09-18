"""A memory written in the model's own coordinates, not an encoder's.

Phase 77 added a memory to the residual stream and nothing happened: own-trajectory
and foreign-trajectory vectors were indistinguishable at every scale and in every
readout. The diagnosis was specific rather than general — the vector was built by
an outside encoder and inserted into a coordinate system the model had built for
itself, so to the model it was structured noise. Activation steering works when
the direction comes from contrast *inside* the model's own states.

There is a direction that is in those coordinates by construction: the gradient of
the loss on the action the agent actually took next, with respect to the hidden
states. One backward pass produces it. It is not fine-tuning — no weight moves —
and it is not an encoding, because nothing outside the model chose its basis.

So this reruns phase 77's one decisive comparison with that change and nothing
else. Store a gradient direction from one position in a trajectory; apply it at a
different position in the same trajectory, and at a position in someone else's;
score the known-good next action both ways.

Falsification and sample size, both fixed before running. The smallest effect
worth acting on is 0.02 NLL — below that a memory is not worth a backward pass.
Phase 77 resolved differences to about ±0.011 at 280 positions, so 200 is the
floor here and 250 is what runs. If own does not beat foreign by an interval that
excludes zero, the channel is closed for gradients too, and the earlier failure
was never about the encoder.
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
    parser.add_argument("--generator", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--trajectories", type=int, default=250)
    parser.add_argument(
        "--per-trajectory",
        type=int,
        default=1,
        help="positions sampled per stream; above one the bootstrap resamples streams, not positions",
    )
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--recent", type=int, default=6)
    parser.add_argument("--gap", type=int, default=4, help="steps between writing and reading")
    parser.add_argument("--scales", default="0.05,0.15,0.3")
    parser.add_argument(
        "--centre",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="remove the direction every stored gradient shares",
    )
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--min-calls", type=int, default=64)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from prepare_claude_tool_stream import open_swe_streams
    from transformers import AutoModelForCausalLM, AutoTokenizer

    streams, _ = open_swe_streams(args.traces, args.min_calls, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.generator)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # float32 throughout: the gradient is the point, and bfloat16 loses too much
    # of a small one to be worth the speed.
    model = AutoModelForCausalLM.from_pretrained(args.generator, dtype=torch.float32).to(
        args.device
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    block = model.model.layers[args.layer]

    bias = {"vector": None, "scale": 0.0}
    captured: dict[str, torch.Tensor] = {}

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if bias["vector"] is not None and bias["scale"]:
            step = bias["vector"].to(hidden.dtype).to(hidden.device)
            hidden = hidden + bias["scale"] * hidden.norm(dim=-1, keepdim=True) * step
        if captured.get("want") is not None:
            # Every parameter is frozen, so nothing upstream carries a graph.
            # Cutting here and making the hidden state its own leaf is enough:
            # the gradient wanted is the one at this layer, not through it.
            hidden = hidden.detach().requires_grad_(True)
            captured["hidden"] = hidden
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    handle = block.register_forward_hook(hook)

    def encode(stream: list[str], position: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        prompt = "Recent steps:\n" + "\n".join(stream[position - args.recent : position])
        prompt += "\n\nWhat is the next step this agent takes?\nNext step:"
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = tokenizer(f" {stream[position].strip()}", add_special_tokens=False)[
            "input_ids"
        ][: args.max_length // 2]
        sequence = [*prompt_ids[-(args.max_length - len(answer_ids)) :], *answer_ids]
        return (
            torch.tensor([sequence], device=args.device),
            torch.tensor([sequence[-len(answer_ids) :]], device=args.device),
            len(answer_ids),
        )

    def score(tokens: torch.Tensor, targets: torch.Tensor, length: int) -> float:
        with torch.inference_mode():
            logits = model(input_ids=tokens).logits[:, -length - 1 : -1]
            return float(
                F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            )

    def gradient_at(stream: list[str], position: int) -> torch.Tensor | None:
        """The direction the loss on this action pushes the hidden states.

        Mean-pooled over the sequence so the stored memory is one vector, the
        same shape phase 77 injected, so nothing but its provenance differs.
        """
        tokens, targets, length = encode(stream, position)
        captured["want"] = True
        captured.pop("hidden", None)
        bias["vector"], bias["scale"] = None, 0.0
        logits = model(input_ids=tokens).logits[:, -length - 1 : -1]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
        captured["want"] = None
        hidden = captured.pop("hidden", None)
        if hidden is None or hidden.grad is None:
            return None
        # Descent direction: the memory should push the way that lowered the loss.
        direction = -hidden.grad[0].mean(0).detach()
        norm = direction.norm()
        return direction / norm if norm > 0 else None

    generator = np.random.default_rng(args.seed)
    usable = [
        index
        for index in range(min(len(streams), args.trajectories))
        if len(streams[index]) > args.recent + args.gap + 2
    ]
    cases = []
    for index in usable:
        stream = streams[index]
        span = len(stream) - args.gap - 1 - args.recent
        wanted = min(args.per_trajectory, max(1, span))
        for write_at in generator.choice(
            range(args.recent, len(stream) - args.gap - 1), size=wanted, replace=False
        ):
            cases.append((index, stream, int(write_at), int(write_at) + args.gap))

    scales = [float(value) for value in args.scales.split(",")]
    records: dict[tuple[str, float], list[float]] = {
        (source, scale): [] for source in ("own", "foreign") for scale in scales
    }
    baseline: list[float] = []
    case_owners: list[int] = []
    memories: dict[int, torch.Tensor] = {}

    for index, stream, write_at, read_at in cases:
        direction = gradient_at(stream, write_at)
        if direction is None:
            continue
        memories[index] = direction

    keys = sorted(memories)
    if len(keys) < 40:
        handle.remove()
        raise RuntimeError(f"only {len(keys)} gradients captured")

    # This codebase already found that mean-pooled states from this model sit in
    # a narrow cone (phase 76: 0.87 average cosine) and centres them everywhere
    # it reads them. Gradients pooled the same way over the same data were never
    # checked. If they share a direction, own and foreign both beating baseline
    # is that shared component, not anything either memory carries.
    stacked = torch.stack([memories[key] for key in keys])
    pairwise = stacked @ stacked.T
    off = (pairwise.sum() - pairwise.diag().sum()) / (len(keys) * (len(keys) - 1))
    print(f"mean pairwise cosine across {len(keys)} stored gradients: {float(off):.4f}")
    if args.centre:
        centred = stacked - stacked.mean(0)
        centred = centred / centred.norm(dim=1, keepdim=True).clamp(min=1e-9)
        after = centred @ centred.T
        off_after = (after.sum() - after.diag().sum()) / (len(keys) * (len(keys) - 1))
        print(f"after centring: {float(off_after):.4f}")
        memories = {key: centred[position] for position, key in enumerate(keys)}

    for offset, (index, stream, write_at, read_at) in enumerate(cases):
        if index not in memories:
            continue
        tokens, targets, length = encode(stream, read_at)
        bias["vector"], bias["scale"] = None, 0.0
        baseline.append(score(tokens, targets, length))
        case_owners.append(index)
        # The foreign memory comes from a trajectory half the corpus away, which
        # is the same pairing phase 77 used.
        other = keys[(keys.index(index) + len(keys) // 2) % len(keys)]
        for source, vector in (("own", memories[index]), ("foreign", memories[other])):
            for scale in scales:
                bias["vector"], bias["scale"] = vector, scale
                records[(source, scale)].append(score(tokens, targets, length))
    handle.remove()

    reference = np.array(baseline)
    owners = case_owners
    print(f"\n{len(reference)} positions, layer {args.layer}, gap {args.gap} steps")
    print(f"no memory: {reference.mean():.4f}\n")
    print(f"{'source':>8} {'scale':>7} {'action NLL':>11} {'gain':>9}")
    for source in ("own", "foreign"):
        for scale in scales:
            values = np.array(records[(source, scale)])
            print(
                f"{source:>8} {scale:7.2f} {values.mean():11.4f}"
                f" {(reference - values).mean():+9.4f}"
            )

    print(f"\n{'comparison':>24} {'difference':>11} {'95% interval':>22}")
    print(f"({len(set(owners))} trajectories, {len(owners)} positions)")
    groups = [np.flatnonzero(np.array(owners) == key) for key in sorted(set(owners))]
    results = {}
    for scale in scales:
        gap = np.array(records[("foreign", scale)]) - np.array(records[("own", scale)])
        maker = np.random.default_rng(7)
        # Cluster bootstrap: resample trajectories whole. Positions inside one
        # stream share its repository, its task and most of its context, so
        # resampling them independently would report an interval narrower than
        # the data supports.
        draws = []
        for _ in range(3000):
            picked = maker.integers(0, len(groups), len(groups))
            draws.append(np.concatenate([groups[i] for i in picked]))
        draws = [gap[rows].mean() for rows in draws]
        low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        results[str(scale)] = {"difference": float(gap.mean()), "low": low, "high": high}
        # 0.02 was named before running as the smallest effect worth a backward pass.
        verdict = (
            "resolved and worth it"
            if low > 0.02
            else "resolved but below the bar"
            if low > 0
            else "NOT resolved"
        )
        print(f"{'own over foreign @ ' + str(scale):>24} {gap.mean():+11.4f}"
              f"  [{low:+.4f}, {high:+.4f}] {verdict}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"positions": len(reference), "own_over_foreign": results}, indent=2)
        )


if __name__ == "__main__":
    main()
