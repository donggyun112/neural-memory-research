from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class LongMemEvalRevisit:
    question_id: str
    question_type: str
    candidates: tuple[str, ...]
    question: str
    answer: str
    target_offset: int
    delay: int
    split: str


def _hash_order(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def _session_text(session: object) -> str:
    if not isinstance(session, list):
        raise ValueError("LongMemEval session must be a list")
    parts: list[str] = []
    for message in session:
        if not isinstance(message, dict):
            raise ValueError("LongMemEval message must be an object")
        role = str(message.get("role", "unknown"))
        content = str(message.get("content", "")).strip()
        if content:
            parts.append(f"[{role}] {content}")
    return "\n".join(parts)


@dataclass(frozen=True)
class LongMemEvalDeferred:
    """One write, one later related event, then a query that arrives last."""

    question_id: str
    question_type: str
    candidates: tuple[str, ...]
    consolidation: str
    question: str
    answer: str
    target_offset: int
    evidence_gap: int
    split: str


def _split_for(question_id: str) -> str:
    return "eval" if int.from_bytes(_hash_order(question_id)[:2]) % 5 == 0 else "train"


def iter_longmemeval_deferred(
    path: Path,
    *,
    candidates: int = 8,
    exclude_abstention: bool = True,
) -> Iterator[LongMemEvalDeferred]:
    """Build episodes where a second evidence session arrives after the write.

    63.8% of non-abstention questions carry evidence in two or more sessions, a
    median of 20 sessions apart.  `iter_longmemeval_revisits` emits one
    single-evidence episode per evidence session and therefore discards that
    structure.  Here the earliest evidence session is the write target, the next
    one becomes a consolidation event that arrives before the query, and every
    evidence session is excluded from the distractor pool so the later event
    cannot be reached as a candidate.
    """
    if candidates < 2:
        raise ValueError("at least two candidates are required")
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("LongMemEval root must be a list")
    for row in payload:
        question_id = str(row["question_id"])
        if exclude_abstention and question_id.endswith("_abs"):
            continue
        session_ids = [str(value) for value in row["haystack_session_ids"]]
        sessions = row["haystack_sessions"]
        if len(session_ids) != len(sessions):
            raise ValueError(f"session id/content mismatch for {question_id}")
        by_id = dict(zip(session_ids, sessions, strict=True))
        position = {value: index for index, value in enumerate(session_ids)}
        evidence = sorted(
            {str(value) for value in row["answer_session_ids"] if str(value) in by_id},
            key=lambda value: position[value],
        )
        if len(evidence) < 2:
            continue
        target_id, consolidation_id = evidence[0], evidence[1]
        evidence_set = set(evidence)
        distractors = [value for value in session_ids if value not in evidence_set]
        if len(distractors) < candidates - 1:
            continue
        ordered = sorted(
            distractors, key=lambda value: _hash_order(f"{question_id}:{value}")
        )[: candidates - 1]
        chosen = [target_id, *ordered]
        chosen.sort(key=lambda value: _hash_order(f"position:{question_id}:{value}"))
        yield LongMemEvalDeferred(
            question_id=question_id,
            question_type=str(row["question_type"]),
            candidates=tuple(_session_text(by_id[value]) for value in chosen),
            consolidation=_session_text(by_id[consolidation_id]),
            question=str(row["question"]),
            answer=str(row.get("answer", "")),
            target_offset=chosen.index(target_id),
            evidence_gap=position[consolidation_id] - position[target_id],
            split=_split_for(question_id),
        )


def iter_longmemeval_revisits(
    path: Path,
    *,
    candidates: int = 8,
    exclude_abstention: bool = True,
) -> Iterator[LongMemEvalRevisit]:
    """Create query-hidden write examples from official evidence-session labels."""
    if candidates < 2:
        raise ValueError("at least two candidates are required")
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("LongMemEval root must be a list")
    for row in payload:
        question_id = str(row["question_id"])
        if exclude_abstention and question_id.endswith("_abs"):
            continue
        session_ids = [str(value) for value in row["haystack_session_ids"]]
        sessions = row["haystack_sessions"]
        if len(session_ids) != len(sessions):
            raise ValueError(f"session id/content mismatch for {question_id}")
        by_id = dict(zip(session_ids, sessions, strict=True))
        answer_ids = [
            str(value) for value in row["answer_session_ids"] if str(value) in by_id
        ]
        answer_set = set(answer_ids)
        distractors = [value for value in session_ids if value not in answer_set]
        if len(distractors) < candidates - 1:
            continue
        split = "eval" if int.from_bytes(_hash_order(question_id)[:2]) % 5 == 0 else "train"
        for answer_id in answer_ids:
            ordered_distractors = sorted(
                distractors,
                key=lambda value: _hash_order(f"{question_id}:{answer_id}:{value}"),
            )[: candidates - 1]
            chosen = [answer_id, *ordered_distractors]
            chosen.sort(key=lambda value: _hash_order(f"position:{question_id}:{answer_id}:{value}"))
            yield LongMemEvalRevisit(
                question_id=question_id,
                question_type=str(row["question_type"]),
                candidates=tuple(_session_text(by_id[value]) for value in chosen),
                question=str(row["question"]),
                answer=str(row.get("answer", "")),
                target_offset=chosen.index(answer_id),
                delay=len(session_ids) - session_ids.index(answer_id),
                split=split,
            )
