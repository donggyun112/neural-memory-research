"""Is the mushroom body's valence anything the model's own entropy does not already give?

This is the fork. `fly-connectome/mushroom_body.py` runs as an organ — sparse code
in, valence out, no retrieval function anywhere — and it generalises to unseen
stimuli for free. But structurally it is another "does this look like something I
have handled before" signal, and that family has already lost here: phase 79's
familiarity filter scored 0.4498 AUC against the entropy of the model's own output
distribution at 0.8179.

So before building anything that gates on valence, the question is whether
valence and entropy are the same number wearing different clothes.

    correlated    entropy already does this, for free, with no organ. The
                  fly direction closes and that is worth knowing.
    uncorrelated  valence carries something the model does not say about
                  itself, which is the first such signal this project has found.

Falsification, fixed before running: if |r| between valence and entropy exceeds
0.5, or valence's AUC against the harder half does not clear entropy's, the
direction is closed. Nothing about a gate is worth building on a signal that is
a noisier copy of one already available.

The organ is taught first, on the same stream, so this measures a mushroom body
that has experience rather than a bare random projection — and the untaught
version is reported alongside, because if teaching changes nothing then the
projection was doing all the work.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fly-connectome"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/open_swe_qwen_stream.pt"))
    parser.add_argument("--traces", type=Path, default=Path("local-data/open-swe-mixed.jsonl"))
    parser.add_argument("--generator", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--taught", type=int, default=2000, help="actions the organ learns from")
    parser.add_argument("--probes", type=int, default=250, help="held-out actions scored")
    parser.add_argument("--active", type=int, default=40)
    parser.add_argument("--recent", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--min-calls", type=int, default=64)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from mushroom_body import MushroomBody, load
    from neural_memory.familiarity import auc
    from prepare_claude_tool_stream import open_swe_streams
    from transformers import AutoModelForCausalLM, AutoTokenizer

    streams, _ = open_swe_streams(args.traces, args.min_calls, args.limit)
    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    offsets = payload["offsets"]
    # Centred, because phase 76 measured these states at 0.87 average cosine and
    # every sparse code would otherwise fire on the shared direction alone.
    turns = F.normalize(payload["turns"] - payload["turns"].mean(0), dim=1).numpy()
    resolved = payload.get("resolved")

    circuit = load(inputs=turns.shape[1], active=args.active, seed=args.seed)
    organ = MushroomBody(circuit)
    untaught = MushroomBody(circuit)

    generator = np.random.default_rng(args.seed)
    usable = [
        index
        for index in range(min(len(streams), len(offsets) - 1))
        if int(offsets[index + 1]) - int(offsets[index]) == len(streams[index])
        and len(streams[index]) > args.recent + 2
    ]
    generator.shuffle(usable)
    cut = len(usable) // 2
    teach_streams, probe_streams = usable[:cut], usable[cut:]

    # The outcome is the trajectory's own: dopamine for a repair that landed,
    # punishment for one that did not. This is the only ground truth the corpus
    # carries, and it is what the organ is built to consume.
    lessons = 0
    for index in teach_streams:
        if lessons >= args.taught:
            break
        outcome = 1.0 if resolved is not None and int(resolved[index]) == 1 else -1.0
        start, stop = int(offsets[index]), int(offsets[index + 1])
        for action in turns[start:stop]:
            organ.teach(action, outcome)
            lessons += 1
            if lessons >= args.taught:
                break
    print(f"organ taught on {lessons} actions; {organ.depleted:.3%} of synaptic weight removed")

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
            action = turns[start + position]

            prompt = "Recent steps:\n" + "\n".join(stream[position - args.recent : position])
            prompt += "\n\nWhat is the next step this agent takes?\nNext step:"
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            answer_ids = tokenizer(f" {stream[position].strip()}", add_special_tokens=False)[
                "input_ids"
            ][: args.max_length // 2]
            if not answer_ids:
                continue
            sequence = [*prompt_ids[-(args.max_length - len(answer_ids)) :], *answer_ids]
            tokens = torch.tensor([sequence], device=args.device)
            logits = model(input_ids=tokens).logits[:, -len(answer_ids) - 1 : -1].float()
            targets = torch.tensor([sequence[-len(answer_ids) :]], device=args.device)
            spread = torch.softmax(logits, dim=-1)
            rows.append(
                {
                    "loss": float(
                        F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                    ),
                    "entropy": float(-(spread * spread.clamp_min(1e-9).log()).sum(-1).mean()),
                    "valence": organ.sense(action),
                    "untaught": untaught.sense(action),
                    "resolved": (
                        int(resolved[index]) if resolved is not None else None
                    ),
                }
            )

    if len(rows) < 40:
        raise RuntimeError(f"only {len(rows)} probes scored")
    loss = np.array([row["loss"] for row in rows])
    entropy = np.array([row["entropy"] for row in rows])
    valence = np.array([row["valence"] for row in rows])
    bare = np.array([row["untaught"] for row in rows])
    hard = loss >= np.median(loss)
    print(f"{len(rows)} held-out actions, median NLL {np.median(loss):.4f}\n")

    correlation = float(np.corrcoef(valence, entropy)[0, 1])
    print(f"{'reader':>16} {'AUC vs hard':>12} {'corr with NLL':>14}")
    scores = {}
    for name, values, sign in (
        ("entropy", entropy, +1.0),
        ("valence", valence, -1.0),
        ("valence untaught", bare, -1.0),
    ):
        # A low valence should mark the harder action, so its sign flips; entropy
        # is read in its natural direction.
        scores[name] = auc(sign * values[hard], sign * values[~hard])
        print(
            f"{name:>16} {scores[name]:12.4f}"
            f" {float(np.corrcoef(values, loss)[0, 1]):14.4f}"
        )
    print(f"{'chance':>16} {0.5:12.4f}")

    print(f"\nvalence against entropy: r = {correlation:+.4f}")
    print(f"teaching moved the reading by r = {float(np.corrcoef(valence, bare)[0, 1]):+.4f} "
          f"against the untaught organ")

    # Phase 79's own writeup kept a caveat rather than arguing it away: scoring a
    # memory signal against next-action likelihood is incoherent, and the honest
    # form is whether the surrounding trajectory actually failed. `resolved` is
    # already loaded for teaching; report the same AUCs against it too, since an
    # "open" verdict built only on the NLL-hardness split inherits that caveat.
    outcomes = np.array(
        [row["resolved"] for row in rows if row["resolved"] is not None]
    )
    if len(outcomes) == len(rows) and 5 <= int((outcomes == 0).sum()) <= len(rows) - 5:
        failed = outcomes == 0
        print(f"\n{'reader':>16} {'AUC vs resolved==0':>19}  (honest form, phase 79's caveat)")
        for name, values, sign in (
            ("entropy", entropy, +1.0),
            ("valence", valence, -1.0),
            ("valence untaught", bare, -1.0),
        ):
            print(
                f"{name:>16} {auc(sign * values[failed], sign * values[~failed]):19.4f}"
            )
    else:
        print(
            "\n(skipping resolved-outcome AUC: labels missing or one class has fewer"
            " than 5 probes)"
        )

    closed = abs(correlation) > 0.5 or scores["valence"] <= scores["entropy"]
    print(
        "\nVERDICT: " + (
            "closed — valence is what entropy already says"
            if closed
            else "open — valence carries something the model does not say about itself"
        )
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "probes": len(rows),
                    "auc": scores,
                    "valence_entropy_r": correlation,
                    "closed": bool(closed),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
