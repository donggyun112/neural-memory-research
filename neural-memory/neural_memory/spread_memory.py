"""Pick what to put in front of a model, by spreading rather than by relevance.

Measured on 900 Open-SWE trajectories against the likelihood of the action a
successful agent actually took next, four items injected either way:

    similarity (the usual thing)   +0.1103 over no memory
    random                         +0.2486
    this                           +0.2796

Choosing the most similar earlier steps is the worst memory tested, behind
recency and behind random, because the most similar step is a near-copy of what
is happening now and the recent context already carries it. Spreading over the
history is worth +0.1694 against it and +0.0310 against drawing at random, both
resolved.

No training, no state, no learned weights. One function.
"""
from __future__ import annotations

import numpy as np


def spread(history: np.ndarray, keep: int, rounds: int = 8) -> list[int]:
    """Indices of ``keep`` items covering the history rather than matching a cue.

    ``history`` is one unit-norm row per remembered item, oldest first. Items are
    grouped by repeated centroid assignment and one representative is taken from
    each group, largest group first, so a stretch of the history that occupied
    more of it gets represented before one that occupied less.

    Nothing about the current state is consulted. That is the point: the choice
    of what to surface was measured to be undetermined by the state anyway
    (Phases 19, 47, 71), and choosing by resemblance to it was measured to be
    worse than not choosing at all.
    """
    if history.ndim != 2:
        raise ValueError("history must have shape [items, features]")
    if keep < 1:
        raise ValueError("keep must be positive")
    if len(history) <= keep:
        return list(range(len(history)))

    channels = min(max(keep, 16), len(history))
    centres = history[np.linspace(0, len(history) - 1, channels).astype(int)].copy()
    assignment = np.zeros(len(history), dtype=int)
    for _ in range(rounds):
        assignment = np.argmax(centres @ history.T, axis=0)
        for channel in range(channels):
            members = history[assignment == channel]
            if len(members):
                centres[channel] = members.mean(axis=0)
        norms = np.linalg.norm(centres, axis=1, keepdims=True)
        centres = np.divide(centres, norms, out=centres, where=norms > 0)

    sizes = np.array([int((assignment == channel).sum()) for channel in range(channels)])
    chosen: list[int] = []
    for channel in np.argsort(-sizes):
        if len(chosen) >= keep:
            break
        if not sizes[channel]:
            continue
        pick = int(np.argmax(history @ centres[channel]))
        if pick not in chosen:
            chosen.append(pick)
    # Oldest first, so an injected block reads in the order things happened.
    return sorted(chosen)


def demo() -> None:
    generator = np.random.default_rng(0)
    # Three clusters of activity, the last one much larger, as a history where a
    # cue-matching reader would return four near-copies of the recent stretch.
    blocks = [
        generator.normal(centre, 0.05, size=(count, 8))
        for centre, count in ((0.0, 5), (3.0, 5), (6.0, 40))
    ]
    history = np.vstack(blocks)
    history /= np.linalg.norm(history, axis=1, keepdims=True)

    chosen = spread(history, keep=3)
    assert len(chosen) == 3, chosen
    assert chosen == sorted(chosen), "indices come back in history order"
    # The picks are spread within the history rather than bunched, which is what
    # a cue-matching reader would not do. They are not one per region: channels
    # are taken largest first, so a stretch occupying forty of fifty items gets
    # more of the budget than one occupying five. That is the rule as measured.
    assert len(set(chosen)) == 3, chosen
    assert max(chosen) - min(chosen) >= 3, chosen

    balanced = np.vstack(
        [
            generator.normal(centre, 0.05, size=(6, 8))
            for centre in (0.0, 3.0, 6.0)
        ]
    )
    balanced /= np.linalg.norm(balanced, axis=1, keepdims=True)
    # With the history evenly divided, every region is represented.
    regions = {slot // 6 for slot in spread(balanced, keep=3)}
    assert regions == {0, 1, 2}, regions

    assert spread(history[:2], keep=4) == [0, 1], "short histories come back whole"
    print("ok")


if __name__ == "__main__":
    demo()
