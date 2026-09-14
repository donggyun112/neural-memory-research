"""Episodes for the question every other builder here throws away.

`iter_longmemeval_deferred`, `_revisits` and `_updates` all pass
`exclude_abstention=True`, so the thirty questions whose answer is "you never
told me that" have never been used. They are the only questions in this corpus
that ask whether the memory holds anything at all, which is exactly what a
familiarity gate is for.

An episode here is one question and its entire real haystack, unmodified: no
candidate sampling, no distractor selection, no target position. The label is
whether the corpus says an answer exists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from neural_memory.longmemeval import _session_text


@dataclass(frozen=True)
class LongMemEvalFamiliarity:
    question_id: str
    question_type: str
    question: str
    answer: str
    sessions: tuple[str, ...]
    answerable: bool
    # Which haystack positions the corpus marks as holding the answer. Removing
    # exactly these makes a matched negative out of an answerable question: same
    # words, same length, same haystack size, evidence gone.
    evidence: tuple[int, ...]


def iter_longmemeval_familiarity(path: Path) -> Iterator[LongMemEvalFamiliarity]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("LongMemEval root must be a list")
    for row in payload:
        question_id = str(row["question_id"])
        sessions = row["haystack_sessions"]
        session_ids = [str(value) for value in row["haystack_session_ids"]]
        if len(session_ids) != len(sessions):
            raise ValueError(f"session id/content mismatch for {question_id}")
        position = {value: index for index, value in enumerate(session_ids)}
        evidence = tuple(
            sorted(
                position[str(value)]
                for value in row["answer_session_ids"]
                if str(value) in position
            )
        )
        yield LongMemEvalFamiliarity(
            question_id=question_id,
            question_type=str(row["question_type"]),
            question=str(row["question"]),
            answer=str(row.get("answer", "")),
            sessions=tuple(_session_text(session) for session in sessions),
            # The abstention questions carry a synthetic answer_session_id that
            # is not in the haystack, so the suffix is the only reliable label.
            answerable=not question_id.endswith("_abs"),
            evidence=evidence,
        )
