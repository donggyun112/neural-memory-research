from __future__ import annotations

import json
from pathlib import Path

from neural_memory.longmemeval import iter_longmemeval_revisits


def test_longmemeval_examples_hide_question_from_write_candidates(tmp_path: Path) -> None:
    sessions = [
        [{"role": "user", "content": f"session {index}"}] for index in range(10)
    ]
    payload = [
        {
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "What was remembered?",
            "answer_session_ids": ["s4"],
            "haystack_session_ids": [f"s{index}" for index in range(10)],
            "haystack_sessions": sessions,
        }
    ]
    path = tmp_path / "longmem.json"
    path.write_text(json.dumps(payload))
    examples = list(iter_longmemeval_revisits(path, candidates=8))
    assert len(examples) == 1
    assert len(examples[0].candidates) == 8
    assert "session 4" in examples[0].candidates[examples[0].target_offset]
    assert all("What was remembered?" not in value for value in examples[0].candidates)
    assert 0 <= examples[0].target_offset < 8
