from __future__ import annotations

import json
from pathlib import Path

from neural_memory.longmemeval import iter_longmemeval_deferred


def _payload(answer_sessions: list[str], sessions: int = 12) -> list[dict[str, object]]:
    return [
        {
            "question_id": "q1",
            "question_type": "multi-session",
            "question": "What was remembered?",
            "answer": "the blue folder",
            "answer_session_ids": answer_sessions,
            "haystack_session_ids": [f"s{index}" for index in range(sessions)],
            "haystack_sessions": [
                [{"role": "user", "content": f"session {index}"}] for index in range(sessions)
            ],
        }
    ]


def _write(tmp_path: Path, payload: list[dict[str, object]]) -> Path:
    path = tmp_path / "longmem.json"
    path.write_text(json.dumps(payload))
    return path


def test_deferred_episode_separates_write_target_from_later_evidence(tmp_path: Path) -> None:
    path = _write(tmp_path, _payload(["s7", "s2"]))
    episodes = list(iter_longmemeval_deferred(path, candidates=8))
    assert len(episodes) == 1
    episode = episodes[0]
    # s2 occurs first in the haystack, so it is the write target and s7 arrives later.
    assert "session 2" in episode.candidates[episode.target_offset]
    assert "session 7" in episode.consolidation
    assert episode.evidence_gap == 5


def test_deferred_episode_keeps_every_evidence_session_out_of_the_distractors(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, _payload(["s2", "s7", "s9"]))
    episode = next(iter_longmemeval_deferred(path, candidates=8))
    others = [
        text for index, text in enumerate(episode.candidates) if index != episode.target_offset
    ]
    for excluded in ("session 7", "session 9"):
        assert all(excluded not in text for text in others)
    assert all(episode.question not in text for text in episode.candidates)


def test_deferred_skips_single_evidence_questions(tmp_path: Path) -> None:
    path = _write(tmp_path, _payload(["s2"]))
    assert list(iter_longmemeval_deferred(path, candidates=8)) == []


def test_deferred_skips_questions_without_enough_distractors(tmp_path: Path) -> None:
    path = _write(tmp_path, _payload(["s0", "s1"], sessions=8))
    assert list(iter_longmemeval_deferred(path, candidates=8)) == []
