from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode LongMemEval at turn granularity, where an answer can fit in one vector"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/longmemeval_s_cleaned.json")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/longmemeval_turns.pt"))
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    rows = [
        row
        for row in json.loads(args.input.read_text())
        if not str(row["question_id"]).endswith("_abs") and str(row.get("answer", "")).strip()
    ]
    generator = np.random.default_rng(args.seed)
    kept = [rows[index] for index in generator.permutation(len(rows))[: args.questions]]

    questions, turns, offsets, targets = [], [], [0], []
    for row in kept:
        answer = str(row["answer"]).strip().lower()
        evidence_ids = {str(value) for value in row["answer_session_ids"]}
        start = len(turns)
        hit = -1
        for session_id, session in zip(
            row["haystack_session_ids"], row["haystack_sessions"], strict=True
        ):
            for message in session:
                content = str(message.get("content", "")).strip()
                if not content:
                    continue
                # The target is the turn that literally contains the gold answer
                # inside a session the corpus marks as evidence. String matching
                # is crude, but it is checkable, and episodes where it finds
                # nothing are dropped rather than guessed at.
                if hit < 0 and str(session_id) in evidence_ids and answer in content.lower():
                    hit = len(turns) - start
                turns.append(f"[{message.get('role', 'unknown')}] {content}")
        if hit < 0:
            del turns[start:]
            continue
        questions.append(str(row["question"]))
        targets.append(hit)
        offsets.append(len(turns))

    if not questions:
        raise RuntimeError("no episode had its answer located in an evidence turn")
    encoder = SentenceTransformer(args.model, device=args.device)
    encoded = encoder.encode(
        [*questions, *turns],
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).cpu()

    output = {
        "source": "LongMemEval-S cleaned, turn granularity",
        "model": args.model,
        "query": encoded[: len(questions)],
        "turns": encoded[len(questions) :],
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"saved questions={len(questions)} of {args.questions} attempted, "
        f"turns={len(turns)}, mean per question={len(turns) / len(questions):.0f}, "
        f"dim={encoded.shape[-1]}, raw_text=false to {args.output}"
    )


if __name__ == "__main__":
    main()
