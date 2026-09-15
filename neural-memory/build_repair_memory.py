"""Turn finished repair runs into the note block a later run is given.

Round one produces, for every task, the agent's own log of what it did and
whether the suite went green. This assembles those into the few lines a round
two agent sees, under the selection rules the earlier phases compared — so the
same conditions that were ranked on next-action likelihood are ranked again on
whether the bug actually gets fixed.

`spread` is the rule phase 72 shipped: cover the history rather than match a
cue, with no selection step. `random` is the control that beat similarity in
phases 73 to 76. `similarity` is the rule everything started with and the one
those phases said is pointed the wrong way. `none` writes nothing.

Only runs that fixed their task contribute. A log of an agent failing is not a
note worth passing on, and phase 74's unresolved split — memory helped the
failures as much as the successes — is exactly what happens when that filter is
missing.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from neural_memory.spread_memory import spread

NOISE = re.compile(r"^(TodoWrite|Task)\b", re.I)


def load_actions(workspace: Path, trials: list[dict], solved_only: bool) -> list[tuple[str, str]]:
    """(task, action) pairs from every run's own log."""
    actions = []
    for trial in trials:
        if solved_only and not trial["resolved"]:
            continue
        log = workspace / trial["task"] / "agent-actions.log"
        if not log.exists():
            continue
        for line in log.read_text().splitlines():
            text = line.strip()
            if text and not NOISE.match(text):
                actions.append((trial["task"], text))
    return actions


def choose(
    condition: str, vectors: np.ndarray, keep: int, cue: np.ndarray | None, seed: int
) -> list[int]:
    if condition == "spread":
        return spread(vectors, keep)
    if condition == "random":
        return sorted(np.random.default_rng(seed).choice(len(vectors), keep, replace=False))
    if condition == "similarity":
        if cue is None:
            raise ValueError("similarity needs a cue")
        return sorted(np.argsort(-(vectors @ cue))[:keep].tolist())
    raise ValueError(f"unknown condition {condition}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True, help="round one outcomes")
    parser.add_argument("--workspace", type=Path, required=True, help="where round one ran")
    parser.add_argument("--condition", choices=("spread", "random", "similarity"), required=True)
    parser.add_argument("--keep", type=int, default=12, help="lines in the note block")
    parser.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--cue", help="text the similarity condition matches against")
    parser.add_argument(
        "--per-task",
        type=Path,
        help="round two tasks; writes one note block per task, cued by its failing tests",
    )
    parser.add_argument("--all-runs", action="store_true", help="include runs that failed")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trials = [json.loads(line) for line in args.trials.read_text().splitlines() if line.strip()]
    actions = load_actions(args.workspace, trials, solved_only=not args.all_runs)
    if len(actions) < args.keep:
        raise RuntimeError(f"only {len(actions)} actions available, need {args.keep}")

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(args.encoder)
    texts = [text for _, text in actions]
    vectors = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    cue = (
        encoder.encode([args.cue], normalize_embeddings=True, show_progress_bar=False)[0]
        if args.cue
        else None
    )

    def write(target: Path, picked: list[int]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(f"- {texts[index]}\n" for index in picked))

    if args.per_task:
        # Similarity picks against the present, so it has to be re-picked for
        # every task rather than chosen once for all of them. The other rules do
        # not depend on a cue and would write the same block each time; they are
        # still written per task so the trial runner reads them the same way.
        targets = [
            json.loads(line)
            for line in args.per_task.read_text().splitlines()
            if line.strip()
        ]
        for task in targets:
            here = (
                encoder.encode(
                    [" ".join(task["failing"][:10])],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0]
                if args.condition == "similarity"
                else cue
            )
            write(args.output / f"{task['name']}.txt", choose(
                args.condition, vectors, args.keep, here, args.seed
            ))
        print(
            f"{args.condition}: {len(targets)} note blocks of {args.keep} lines"
            f" from {len(actions)} actions in {len(trials)} runs -> {args.output}/"
        )
        return

    picked = choose(args.condition, vectors, args.keep, cue, args.seed)
    write(args.output, picked)
    sources = {actions[index][0] for index in picked}
    print(
        f"{args.condition}: {len(picked)} lines from {len(sources)} tasks"
        f" out of {len(actions)} actions in {len(trials)} runs -> {args.output}"
    )


def demo() -> None:
    """Each rule must return the requested count without repeating an index."""
    vectors = np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    cue = vectors[3]
    for condition in ("spread", "random", "similarity"):
        picked = choose(condition, vectors, 5, cue, seed=1)
        assert len(picked) == 5 and len(set(picked)) == 5, (condition, picked)
    # Similarity must return the cue's own row first; that it does is why the
    # earlier phases could show the choice duplicates what is already in context.
    assert 3 in choose("similarity", vectors, 5, cue, seed=1)
    print("demo ok")


if __name__ == "__main__":
    main()
