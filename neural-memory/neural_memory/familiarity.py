"""A fixed-size familiarity signal that lives on cells, not on items.

Phase 31 established that a content store cannot beat cosine at retrieval,
because a linear associative read is at best the similarity structure it was
handed. This is a different quantity. Marking the cells a document activates,
and later asking what fraction of a cue's cells are already marked, measures
accumulated evidence across *every* stored item at once. Maximum cosine measures
the single best match. Neither approximates the other:

    cue A matches one stored item at 0.6           -> max cosine says A
    cue B matches twenty stored items at 0.3 each  -> cell overlap says B

The fly-connectome line's phase 37 showed this mechanism is exactly a Bloom
filter over cell identities, with a capacity predictable from the number of
cells and the code width. What it buys here is that the state is fixed size:
it does not grow with the number of documents written, and it needs no decision
about how many matches to aggregate.
"""
from __future__ import annotations

import numpy as np


def sparse_code(features: np.ndarray, projection: np.ndarray, active: int) -> np.ndarray:
    """Expand into cell space and keep the strongest cells, renormalised.

    Rows are returned with the inactive cells zeroed, so a code is both the
    address and the weighting the readout uses.
    """
    if active < 1:
        raise ValueError("at least one cell must stay active")
    expanded = np.maximum(0.0, features @ projection)
    if active >= expanded.shape[-1]:
        kept = expanded
    else:
        cut = np.partition(expanded, -active, axis=-1)[..., -active][..., None]
        kept = np.where(expanded >= cut, expanded, 0.0)
    norm = np.linalg.norm(kept, axis=-1, keepdims=True)
    return np.divide(kept, norm, out=np.zeros_like(kept), where=norm > 0)


class FamiliarityFilter:
    """Marks cells as they are used, and reports how much of a cue is known.

    ``graded`` decides what a cell remembers. Clamped, it holds one bit and the
    store is the Bloom filter the fly line's phase 37 identified. Graded, it
    sums the activation each document contributed, which keeps how *strongly*
    the cells were driven and not only that they were. The two separate whether
    a fixed-size accumulator fails because it merges documents together or
    because it throws away their magnitudes.
    """

    def __init__(self, cells: int, decay: float = 1.0, graded: bool = False) -> None:
        if cells < 1:
            raise ValueError("the filter needs at least one cell")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must lie in (0, 1]")
        self.mark = np.zeros(cells)
        self.decay = decay
        self.graded = graded

    def write(self, code: np.ndarray) -> None:
        if code.shape != self.mark.shape:
            raise ValueError("code and filter must span the same cells")
        if self.decay < 1.0:
            self.mark *= self.decay
        if self.graded:
            self.mark = self.mark + code
        else:
            self.mark = np.maximum(self.mark, (code > 0).astype(float))

    def score(self, code: np.ndarray) -> float:
        """Share of the cue's own weight that lands on already-marked cells.

        Weighting by the cue's own activation rather than counting cells keeps
        the signal sensitive to which of its cells are known, not merely how
        many.
        """
        if code.shape != self.mark.shape:
            raise ValueError("code and filter must span the same cells")
        total = float(code.sum())
        if total <= 0.0:
            return 0.0
        return float((code * self.mark).sum() / total)


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability a positive scores above a negative; ties count as half."""
    if positive.size == 0 or negative.size == 0:
        raise ValueError("both groups must be non-empty")
    difference = positive[:, None] - negative[None, :]
    return float((difference > 0).mean() + 0.5 * (difference == 0).mean())
