"""When the answer needs two sessions, can a store reach the one the question cannot?

Phase 31 measured retrieval as picking a single evidence session and found the
associative store beaten by cosine. That task has no composition in it. On
LongMemEval's multi-evidence questions the structure is different, and a
diagnostic run before this one established it: the harder evidence session sits
in the top four for only 0.176 of the questions where it starts outside, but
reaches 0.676 once the cue is expanded with the easier evidence session. The
session the question cannot find is reachable through the session it can.

That is what a superposed associative read does in one matrix multiply: the read
is a sum of stored values weighted by key similarity, so anything close to what
the cue retrieves is pulled along with it. Phase 31 called that smearing and
counted it against the store. On a single target it is noise. On a set it may be
the mechanism.

Readers, fixed before measuring:

  cosine            single-shot cosine to the question, top-k. The standard.
  prf               pseudo-relevance feedback: take the top-1, add it to the
                    cue, retrieve again. Forty years old, O(N), and the real
                    baseline any second-hop method has to beat.
  prf_oracle        the same with the true easy evidence session as the first
                    hop. Not a competitor, a ceiling on what one hop can buy.
  store             the delta-rule associative read Phase 31 rejected.

Every reader returns exactly k sessions, so none can win by returning more.
Falsification: if the store does not beat prf, the superposition adds nothing
that explicit expansion does not already do more cheaply.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neural_memory.familiarity import sparse_code


def delta_store(keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Error-correcting write of every session, in order."""
    matrix = np.zeros((values.shape[1], keys.shape[1]))
    for key, value in zip(keys, values):
        matrix = matrix + np.outer(value - matrix @ key, key)
    return matrix


def recall_at(chosen: np.ndarray, evidence: np.ndarray) -> float:
    """Share of the evidence set the reader returned."""
    return float(np.isin(evidence, chosen).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_familiarity.pt")
    )
    parser.add_argument("--cells", type=int, default=2048)
    parser.add_argument("--active", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=256)
    parser.add_argument("--keeps", default="2,4,8")
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    queries = torch.nn.functional.normalize(payload["query"], dim=-1).numpy()
    sessions = torch.nn.functional.normalize(payload["sessions"], dim=-1).numpy()
    offsets = payload["offsets"].numpy()
    evidence_flat = payload["evidence"].numpy()
    evidence_offsets = payload["evidence_offsets"].numpy()
    answerable = payload["answerable"].numpy()
    keeps = [int(value) for value in args.keeps.split(",")]

    episodes = [
        index
        for index in range(len(queries))
        if answerable[index]
        and evidence_offsets[index + 1] - evidence_offsets[index] >= 2
    ]
    if not episodes:
        raise RuntimeError("no question carries two locatable evidence sessions")

    names = ["cosine", "prf", "prf_soft", "prf_gated", "prf_oracle", "store"]
    scores = {
        (name, keep): np.zeros((len(args.seeds.split(",")), len(episodes)))
        for name in names
        for keep in keeps
    }
    for row, seed in enumerate(int(value) for value in args.seeds.split(",")):
        generator = np.random.default_rng(seed)
        key_projection = generator.standard_normal(
            (queries.shape[-1], args.cells)
        ) / np.sqrt(queries.shape[-1])
        value_projection = generator.standard_normal(
            (queries.shape[-1], args.value_dim)
        ) / np.sqrt(queries.shape[-1])
        for column, index in enumerate(episodes):
            block = sessions[offsets[index] : offsets[index + 1]]
            evidence = evidence_flat[evidence_offsets[index] : evidence_offsets[index + 1]]
            cue = queries[index]
            to_question = block @ cue

            keys = sparse_code(block, key_projection, args.active)
            values = np.tanh(block @ value_projection)
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            read = delta_store(keys, values) @ sparse_code(
                cue[None], key_projection, args.active
            )[0]
            norm = np.linalg.norm(read)
            store_scores = values @ (read / norm) if norm > 0 else np.zeros(len(block))

            first = int(np.argmax(to_question))
            expanded = cue + block[first]
            prf_scores = block @ (expanded / np.linalg.norm(expanded))

            # A single first hop is all-or-nothing, and the oracle gap says a
            # wrong one costs more than a right one gains. Averaging the top few,
            # weighted by their own similarity, dilutes a bad hop instead of
            # committing to it.
            order_now = np.argsort(-to_question)[:4]
            weights = np.maximum(0.0, to_question[order_now])
            if weights.sum() > 0:
                blend = cue + (weights[:, None] * block[order_now]).sum(axis=0) / weights.sum()
            else:
                blend = cue
            soft_scores = block @ (blend / np.linalg.norm(blend))

            # Expand only when the top hit stands clearly above the next one,
            # which is the cheapest available signal that the hop is worth taking.
            ordered_now = np.sort(to_question)[::-1]
            confident = float(ordered_now[0] - ordered_now[1]) > 0.02
            gated_scores = prf_scores if confident else to_question
            easy = min(evidence, key=lambda slot: -to_question[int(slot)])
            oracle_cue = cue + block[int(easy)]
            oracle_scores = block @ (oracle_cue / np.linalg.norm(oracle_cue))

            for name, value in (
                ("cosine", to_question),
                ("prf", prf_scores),
                ("prf_soft", soft_scores),
                ("prf_gated", gated_scores),
                ("prf_oracle", oracle_scores),
                ("store", store_scores),
            ):
                order = np.argsort(-value)
                for keep in keeps:
                    scores[(name, keep)][row, column] = recall_at(order[:keep], evidence)

    results: dict[str, dict[str, float]] = {}
    for (name, keep), block in scores.items():
        per_question = block.mean(axis=0)
        generator = np.random.default_rng(7)
        resampled = [
            per_question[generator.integers(0, len(per_question), len(per_question))].mean()
            for _ in range(3000)
        ]
        results[f"{name}@{keep}"] = {
            "recall": float(per_question.mean()),
            "low": float(np.percentile(resampled, 2.5)),
            "high": float(np.percentile(resampled, 97.5)),
        }

    print(f"{len(episodes)} questions needing two or more evidence sessions")
    print("Share of the evidence set returned. Every reader returns the same count.\n")
    print(f"{'reader':>12}" + ''.join(f"{'@' + str(keep):>22}" for keep in keeps))
    for name in names:
        cells = ''.join(
            f"{results[f'{name}@{keep}']['recall']:10.4f}"
            f" [{results[f'{name}@{keep}']['low']:.3f},{results[f'{name}@{keep}']['high']:.3f}]"
            for keep in keeps
        )
        print(f"{name:>12}{cells}")

    print(f"\n{'comparison':>34} {'difference':>11} {'95% interval':>22}")
    head_to_head = {}
    for keep in keeps:
        for left, right in (
            ("prf_soft", "cosine"),
            ("prf_soft", "prf"),
            ("prf_gated", "cosine"),
            ("store", "cosine"),
        ):
            gap = scores[(left, keep)].mean(axis=0) - scores[(right, keep)].mean(axis=0)
            generator = np.random.default_rng(7)
            resampled = [
                gap[generator.integers(0, len(gap), len(gap))].mean() for _ in range(3000)
            ]
            low, high = (
                float(np.percentile(resampled, 2.5)),
                float(np.percentile(resampled, 97.5)),
            )
            label = f"{left} minus {right} @{keep}"
            head_to_head[label] = {"difference": float(gap.mean()), "low": low, "high": high}
            resolved = "resolved" if low > 0 or high < 0 else "NOT resolved"
            print(f"{label:>34} {gap.mean():+11.4f}  [{low:+.4f}, {high:+.4f}] {resolved}")

    output = {
        "config": vars(args),
        "questions": len(episodes),
        "results": results,
        "head_to_head": head_to_head,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
