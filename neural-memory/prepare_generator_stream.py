"""Re-encode an action stream in the acting model's own representation.

Everything measured so far put the memory in BGE-small's space: 33 million
parameters trained for sentence similarity, deciding what a 1.5-billion
parameter model should be shown. The model that knows what a shell command does
was not part of choosing.

That is not a philosophical complaint, it is the direct reading of Phase 73 and
74: similarity in BGE space is *worse than random*, so that space
anti-correlates with usefulness here. Two `ls` calls on different directories
look nearly identical to it and are functionally unrelated; a test failure and
the edit that fixes it look unrelated and are causally joined. The fly makes the
same point structurally — KC to MBON is part of the circuit that computes the
behaviour, not a separate module feeding it.

This encodes each action with the generator's own hidden states instead, so
similarity means "alike to the model that will act" rather than "alike in
words". Same artifact shape as the BGE streams, so every evaluation runs on it
unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from prepare_claude_tool_stream import open_swe_outcomes, open_swe_streams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("local-data/open-swe-mixed.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/open_swe_qwen_stream.pt")
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--min-calls", type=int, default=64)
    parser.add_argument("--limit", type=int, default=400, help="streams to encode")
    parser.add_argument("--layer", type=int, default=-1, help="hidden layer to read")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    streams, failures = open_swe_streams(args.root, args.min_calls, args.limit)
    outcomes = open_swe_outcomes(args.root, args.min_calls, args.limit)
    if not streams:
        raise RuntimeError("no stream met the minimum call count")
    flat = [text for stream in streams for text in stream]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), output_hidden_states=True
    ).to(args.device)
    model.eval()

    vectors = []
    with torch.inference_mode():
        for start in range(0, len(flat), args.batch_size):
            batch = flat[start : start + args.batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(args.device)
            hidden = model(**encoded).hidden_states[args.layer]
            # Mean over real tokens only; padding would drag every short action
            # toward the same point.
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors.append(F.normalize(pooled.float(), dim=-1).cpu())
            if start % (args.batch_size * 100) == 0:
                print(f"encoded {start}/{len(flat)}", flush=True)

    encoded_all = torch.cat(vectors)
    offsets = [0]
    for stream in streams:
        offsets.append(offsets[-1] + len(stream))
    output = {
        "source": f"{args.root.name}, encoded by {args.model} layer {args.layer}",
        "model": args.model,
        "turns": encoded_all,
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "targets": torch.full((len(streams),), -1, dtype=torch.long),
        "failed": torch.tensor(
            [flag for stream in failures for flag in stream], dtype=torch.bool
        ),
        "resolved": torch.tensor(outcomes or [-1] * len(streams), dtype=torch.long),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"saved sessions={len(streams)}, actions={len(flat)}, "
        f"dim={encoded_all.shape[-1]}, raw_text=false to {args.output}"
    )


if __name__ == "__main__":
    main()
