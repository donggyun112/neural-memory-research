from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neural_memory.familiarity import FamiliarityFilter, auc, sparse_code
from neural_memory.longmemeval_familiarity import iter_longmemeval_familiarity


def test_sparse_code_keeps_a_budget_and_normalises() -> None:
    coded = sparse_code(np.array([[1.0, -2.0, 0.5, 3.0]]), np.eye(4), 2)
    assert int((coded != 0).sum()) == 2
    assert float(np.linalg.norm(coded)) == pytest.approx(1.0)


def test_sparse_code_rejects_an_empty_budget() -> None:
    with pytest.raises(ValueError):
        sparse_code(np.zeros((1, 4)), np.eye(4), 0)


def test_an_empty_filter_recognises_nothing() -> None:
    memory = FamiliarityFilter(4)
    assert memory.score(np.array([0.5, 0.5, 0.0, 0.0])) == 0.0


def test_a_written_code_reads_as_fully_familiar() -> None:
    memory = FamiliarityFilter(4)
    code = np.array([0.6, 0.8, 0.0, 0.0])
    memory.write(code)
    assert memory.score(code) == pytest.approx(1.0)


def test_a_disjoint_cue_reads_as_unfamiliar() -> None:
    memory = FamiliarityFilter(4)
    memory.write(np.array([1.0, 0.0, 0.0, 0.0]))
    assert memory.score(np.array([0.0, 0.0, 0.6, 0.8])) == pytest.approx(0.0)


def test_the_binary_filter_forgets_how_often_a_cell_was_used() -> None:
    once, twice = FamiliarityFilter(4), FamiliarityFilter(4)
    code = np.array([1.0, 0.0, 0.0, 0.0])
    once.write(code)
    twice.write(code)
    twice.write(code)
    assert np.array_equal(once.mark, twice.mark)


def test_the_graded_filter_keeps_how_strongly_cells_were_driven() -> None:
    # This is the whole difference the paired evaluation turns on: a binary mark
    # cannot tell one weak document from ten strong ones on the same cell.
    weak, strong = FamiliarityFilter(4, graded=True), FamiliarityFilter(4, graded=True)
    weak.write(np.array([0.1, 0.0, 0.0, 0.0]))
    for _ in range(10):
        strong.write(np.array([1.0, 0.0, 0.0, 0.0]))
    assert float(strong.mark[0]) > float(weak.mark[0])


def test_the_filter_rejects_a_code_of_the_wrong_width() -> None:
    with pytest.raises(ValueError):
        FamiliarityFilter(4).write(np.zeros(3))


def test_auc_is_half_for_identical_groups() -> None:
    assert auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(0.5)


def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "multi-session",
                    "question": "where?",
                    "answer": "there",
                    "haystack_session_ids": ["s0", "s1", "s2"],
                    "haystack_sessions": [
                        [{"role": "user", "content": f"session {index}"}] for index in range(3)
                    ],
                    "answer_session_ids": ["s2"],
                },
                {
                    "question_id": "q2_abs",
                    "question_type": "multi-session",
                    "question": "who?",
                    "answer": "you never said",
                    "haystack_session_ids": ["s0"],
                    "haystack_sessions": [[{"role": "user", "content": "session 0"}]],
                    "answer_session_ids": ["answer_q2_abs"],
                },
            ]
        )
    )
    return path


def test_abstention_questions_are_labelled_and_carry_no_evidence(tmp_path: Path) -> None:
    episodes = list(iter_longmemeval_familiarity(_corpus(tmp_path)))
    answerable, abstention = episodes
    assert answerable.answerable and answerable.evidence == (2,)
    # The abstention answer_session_id is synthetic and not in the haystack, so
    # no position can be located for it.
    assert not abstention.answerable and abstention.evidence == ()


def test_every_haystack_session_survives_into_the_episode(tmp_path: Path) -> None:
    episode = next(iter_longmemeval_familiarity(_corpus(tmp_path)))
    assert len(episode.sessions) == 3
    assert episode.sessions[2] == "[user] session 2"
