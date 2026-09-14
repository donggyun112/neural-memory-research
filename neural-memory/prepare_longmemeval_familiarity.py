from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from neural_memory.longmemeval_familiarity import iter_longmemeval_familiarity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode whole haystacks for the abstention/answerable gate test"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("artifacts/longmemeval_s_cleaned.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/longmemeval_familiarity.pt")
    )
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--answerable", type=int, default=200, help="answerable questions to keep")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    episodes = list(iter_longmemeval_familiarity(args.input))
    abstention = [episode for episode in episodes if not episode.answerable]
    answerable = [episode for episode in episodes if episode.answerable]
    if not abstention:
        raise RuntimeError("no abstention questions were found")
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(answerable), generator=generator)[: args.answerable]
    kept = [*abstention, *(answerable[int(index)] for index in order)]

    texts = [text for episode in kept for text in (episode.question, *episode.sessions)]
    encoder = SentenceTransformer(args.model, device=args.device)
    encoded = encoder.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).cpu()

    questions, sessions, offsets = [], [], [0]
    cursor = 0
    for episode in kept:
        questions.append(encoded[cursor])
        sessions.append(encoded[cursor + 1 : cursor + 1 + len(episode.sessions)])
        cursor += 1 + len(episode.sessions)
        offsets.append(offsets[-1] + len(episode.sessions))
    output = {
        "source": "LongMemEval-S cleaned, whole haystacks",
        "model": args.model,
        "query": torch.stack(questions),
        # Haystacks differ in length, so sessions are stored flat with offsets
        # rather than padded into a rectangle that would invent empty documents.
        "sessions": torch.cat(sessions),
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "answerable": torch.tensor(
            [episode.answerable for episode in kept], dtype=torch.bool
        ),
        # Question length is not matched between the two groups in this corpus,
        # so it travels with the features as a baseline rather than a surprise.
        "question_chars": torch.tensor(
            [float(len(episode.question)) for episode in kept]
        ),
        # Evidence positions, flat with their own offsets, so a matched negative
        # can be built by writing the same haystack minus exactly these.
        "evidence": torch.tensor(
            [position for episode in kept for position in episode.evidence], dtype=torch.long
        ),
        "evidence_offsets": torch.tensor(
            np.cumsum([0, *(len(episode.evidence) for episode in kept)]).tolist(),
            dtype=torch.long,
        ),
        "question_type_names": sorted({episode.question_type for episode in kept}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"saved questions={len(kept)} (abstention={len(abstention)}), "
        f"sessions={output['sessions'].shape[0]}, dim={encoded.shape[-1]}, "
        f"raw_text=false to {args.output}"
    )


if __name__ == "__main__":
    main()
