from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from analyze_knowledge_update import key_overlap, read_scores, write_store
from neural_memory.longmemeval import iter_longmemeval_updates


def _corpus(tmp_path: Path, question_type: str = "knowledge-update") -> Path:
    sessions = [[{"role": "user", "content": f"session {index}"}] for index in range(5)]
    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": question_type,
                    "question": "what now?",
                    "answer": "the new one",
                    "haystack_session_ids": [f"s{index}" for index in range(5)],
                    "haystack_sessions": sessions,
                    # Out of order on purpose: the builder must sort by position.
                    "answer_session_ids": ["s3", "s1"],
                }
            ]
        )
    )
    return path


def test_the_target_is_the_later_statement_and_both_stay_in_the_pool(tmp_path: Path) -> None:
    episode = next(iter_longmemeval_updates(_corpus(tmp_path), candidates=4))
    assert episode.candidates[episode.target_offset] == "[user] session 3"
    assert episode.candidates[episode.superseded_offset] == "[user] session 1"
    assert episode.target_offset > episode.superseded_offset


def test_other_question_types_are_skipped(tmp_path: Path) -> None:
    assert list(iter_longmemeval_updates(_corpus(tmp_path, "multi-session"), candidates=4)) == []


def test_too_few_candidates_to_hold_two_statements_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        next(iter_longmemeval_updates(_corpus(tmp_path), candidates=2))


def test_a_later_write_on_shared_units_replaces_the_earlier_value() -> None:
    torch.manual_seed(3)
    key = F.normalize(torch.zeros(8).index_fill_(0, torch.tensor([0, 1, 2]), 1.0), dim=0)
    values = F.normalize(torch.randn(2, 4), dim=1)
    # Both statements land on exactly the same units, which is the case the
    # delta rule is supposed to handle by overwriting rather than accumulating.
    state = write_store(torch.stack([key, key]), values)
    scores = read_scores(state, key, values)
    assert float(scores[1]) > float(scores[0])


def test_key_overlap_counts_shared_active_units() -> None:
    keys = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    assert key_overlap(keys, 0, 1) == pytest.approx(0.5)
