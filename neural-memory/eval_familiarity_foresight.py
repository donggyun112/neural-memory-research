"""Does a fixed-size familiarity signal know where the model will do badly?

Everything this project added to a model failed for one reason: it carried
information the model already had. Retrieval returns what the context already
says, and phase 75 showed a better encoder finds a closer duplicate and pays more
for it. An injected activation meant nothing in the model's own coordinates.

Familiarity is not in that category. It does not return content; it reports
coverage. A model cannot ask itself whether a pattern is new, because its own
confidence is token likelihood — fluency, not possession — so the signal is
genuinely absent at decision time. It is also the one mechanism here that
measured well: 0.97 win rate at telling a supported cue from an unsupported one,
in O(1) state.

The claim is narrow and has one gate. Write the hidden states the model has
already seen into a Bloom-style filter, and ask whether a held-out state's
overlap predicts that the model will predict its continuation badly. It only
counts if it beats the trivial readers, because phase 12 was lost to
`question_length`: a signal that merely tracks how long or how uncertain the
input is has been reinvented, not found.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from neural_memory.familiarity import FamiliarityFilter, auc, sparse_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/open_swe_qwen_stream.pt"))
    parser.add_argument("--traces", type=Path, default=Path("local-data/open-swe-mixed.jsonl"))
    parser.add_argument("--generator", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--written", type=int, default=3000, help="states written to the filter")
    parser.add_argument("--probes", type=int, default=400, help="held-out actions scored")
    parser.add_argument("--cells", type=int, default=4096)
    parser.add_argument("--active", type=int, default=64)
    parser.add_argument("--recent", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--min-calls", type=int, default=64)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from prepare_claude_tool_stream import open_swe_streams
    from transformers import AutoModelForCausalLM, AutoTokenizer

    streams, _ = open_swe_streams(args.traces, args.min_calls, args.limit)
    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    turns = payload["turns"]
    offsets = payload["offsets"]
    # Mean-pooled hidden states sit in a narrow cone (phase 76 measured 0.87
    # average cosine), where every code would overlap every other and the filter
    # would saturate on the shared direction alone.
    turns = F.normalize(turns - turns.mean(0), dim=1)

    generator = np.random.default_rng(args.seed)
    projection = generator.normal(size=(turns.shape[1], args.cells)) / np.sqrt(turns.shape[1])

    # Streams are split whole: writing part of a trajectory and probing the rest
    # would measure how repetitive one agent is, not what the filter has seen.
    usable = [
        index
        for index in range(min(len(streams), len(offsets) - 1))
        if int(offsets[index + 1]) - int(offsets[index]) == len(streams[index])
        and len(streams[index]) > args.recent + 2
    ]
    generator.shuffle(usable)
    cut = len(usable) // 2
    written_streams, probe_streams = usable[:cut], usable[cut:]

    filter_ = FamiliarityFilter(args.cells, graded=True)
    seen = 0
    for index in written_streams:
        start, stop = int(offsets[index]), int(offsets[index + 1])
        block = turns[start:stop].numpy()
        for code in sparse_code(block, projection, args.active):
            filter_.write(code)
            seen += 1
            if seen >= args.written:
                break
        if seen >= args.written:
            break
    print(f"{seen} states written to a {args.cells}-cell filter")

    tokenizer = AutoTokenizer.from_pretrained(args.generator)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.generator, dtype=getattr(torch, args.dtype)
    ).to(args.device)
    model.eval()

    rows: list[dict] = []
    with torch.inference_mode():
        for index in probe_streams:
            if len(rows) >= args.probes:
                break
            stream = streams[index]
            start = int(offsets[index])
            position = int(generator.integers(args.recent, len(stream) - 1))
            code = sparse_code(turns[start + position].numpy()[None], projection, args.active)[0]

            prompt = "Recent steps:\n" + "\n".join(stream[position - args.recent : position])
            prompt += "\n\nWhat is the next step this agent takes?\nNext step:"
            answer = stream[position].strip()
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            answer_ids = tokenizer(f" {answer}", add_special_tokens=False)["input_ids"]
            answer_ids = answer_ids[: args.max_length // 2]
            if not answer_ids:
                continue
            sequence = [*prompt_ids[-(args.max_length - len(answer_ids)) :], *answer_ids]
            tokens = torch.tensor([sequence], device=args.device)
            logits = model(input_ids=tokens).logits[:, -len(answer_ids) - 1 : -1].float()
            targets = torch.tensor([sequence[-len(answer_ids) :]], device=args.device)
            loss = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)))
            # The trivial readers, measured on the same forward pass so neither
            # gets an advantage of timing.
            spread = torch.softmax(logits, dim=-1)
            entropy = float(-(spread * spread.clamp_min(1e-9).log()).sum(-1).mean())
            rows.append(
                {
                    "loss": loss,
                    "familiarity": filter_.score(code),
                    "entropy": entropy,
                    "length": float(len(answer_ids)),
                    "prompt_length": float(len(prompt_ids)),
                }
            )

    if len(rows) < 40:
        raise RuntimeError(f"only {len(rows)} probes scored")
    loss = np.array([row["loss"] for row in rows])
    hard = loss >= np.median(loss)
    print(f"{len(rows)} held-out actions, median NLL {np.median(loss):.4f}\n")

    print(f"{'reader':>16} {'state':>7} {'AUC vs hard':>12} {'|corr|':>8}")
    results: dict[str, dict] = {}
    for name, state, sign in (
        ("familiarity", "O(1)", -1.0),
        ("entropy", "none", +1.0),
        ("answer_length", "none", +1.0),
        ("prompt_length", "none", +1.0),
    ):
        key = {"answer_length": "length"}.get(name, name)
        values = np.array([row[key] for row in rows])
        # A low-familiarity action should be the hard one, so the sign flips for
        # it; the others are read in their natural direction.
        score = auc(sign * values[hard], sign * values[~hard])
        correlation = float(np.corrcoef(values, loss)[0, 1])
        results[name] = {"auc": score, "corr": correlation}
        print(f"{name:>16} {state:>7} {score:12.4f} {abs(correlation):8.4f}")
    print(f"{'chance':>16} {'':>7} {0.5:12.4f}")

    print(f"\n{'comparison':>36} {'difference':>11}")
    for control in ("entropy", "answer_length", "prompt_length"):
        gap = results["familiarity"]["auc"] - results[control]["auc"]
        verdict = "familiarity ahead" if gap > 0 else "the trivial reader ahead"
        print(f"{'familiarity over ' + control:>36} {gap:+11.4f}  {verdict}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"probes": len(rows), "readers": results}, indent=2))


def demo() -> None:
    """A reader that only tracks length must not be mistaken for a signal."""
    rows = np.array([1.0, 2.0, 3.0, 4.0])
    hard = rows >= np.median(rows)
    assert auc(-rows[hard], -rows[~hard]) == 0.0, "a perfectly wrong reader scores zero"
    assert auc(rows[hard], rows[~hard]) == 1.0
    filter_ = FamiliarityFilter(16, graded=True)
    projection = np.random.default_rng(0).normal(size=(8, 16))
    codes = sparse_code(np.random.default_rng(1).normal(size=(3, 8)), projection, 4)
    filter_.write(codes[0])
    # Something written must score above something never seen.
    assert filter_.score(codes[0]) > filter_.score(codes[2]) or filter_.score(codes[2]) == 0.0
    print("demo ok")


if __name__ == "__main__":
    main()
