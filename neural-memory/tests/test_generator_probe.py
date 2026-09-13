from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from eval_generator_utility import condition_indices, length_matched_selection
from neural_memory.generator_probe import answer_token_nll, build_memory_prompt
from neural_memory.longmemeval import iter_longmemeval_revisits


def test_build_memory_prompt_orders_and_truncates_sessions() -> None:
    prompt = build_memory_prompt(["a" * 10, "  ", "b" * 10], "why?", max_session_chars=4)
    assert "Memory 1:\naaaa\n\nMemory 2:\nbbbb" in prompt
    assert prompt.endswith("Question: why?\nAnswer:")


def test_build_memory_prompt_without_memory_keeps_only_the_question() -> None:
    assert build_memory_prompt([], "why?") == "Question: why?\nAnswer:"


def test_answer_token_nll_scores_only_answer_positions() -> None:
    input_ids = torch.tensor([[5, 6, 7]])
    answer_mask = torch.tensor([[0, 0, 1]])
    logits = torch.zeros(1, 3, 8)
    # Position 1 predicts token 7; make it certain and leave prompt positions wrong.
    logits[0, 1, 7] = 20.0
    logits[0, 0, 6] = -20.0
    assert float(answer_token_nll(logits, input_ids, answer_mask)) == pytest.approx(0.0, abs=1e-4)


def test_answer_token_nll_requires_a_scored_token() -> None:
    with pytest.raises(ValueError):
        answer_token_nll(torch.zeros(1, 3, 8), torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long))


def test_condition_indices_respect_capacity_and_gold_slot() -> None:
    learned = torch.tensor([[True, False, False, True]])
    targets = torch.tensor([2])
    conditions = condition_indices(learned, targets, traces=4, keep=2, seed=7)
    assert conditions["learned"] == [[0, 3]]
    assert conditions["recency"] == [[2, 3]]
    assert conditions["no_memory"] == [[]]
    assert conditions["all_traces"] == [[0, 1, 2, 3]]
    assert len(conditions["random"][0]) == 2
    # The oracle holds the gold slot at the same capacity as every other condition.
    assert 2 in conditions["oracle"][0]
    assert len(conditions["oracle"][0]) == 2


def test_length_matched_selection_tracks_total_length_without_copying() -> None:
    lengths = [[100, 900, 500, 500]]
    learned = [[0, 1]]
    matched = length_matched_selection(lengths, learned, 2)
    # The learned pair totals 1000; slots 2 and 3 are the closest different pair.
    assert matched == [[2, 3]]


def test_longmemeval_examples_carry_the_gold_answer(tmp_path: Path) -> None:
    payload = [
        {
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "What was remembered?",
            "answer": "the blue folder",
            "answer_session_ids": ["s4"],
            "haystack_session_ids": [f"s{index}" for index in range(10)],
            "haystack_sessions": [
                [{"role": "user", "content": f"session {index}"}] for index in range(10)
            ],
        }
    ]
    path = tmp_path / "longmem.json"
    path.write_text(json.dumps(payload))
    examples = list(iter_longmemeval_revisits(path, candidates=8))
    assert examples[0].answer == "the blue folder"
