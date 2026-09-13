from __future__ import annotations

import json
from pathlib import Path

import torch

from combine_context_features import combine_payloads
from neural_memory.context_memory import (
    DualEncoderRecallMemory,
    MultiTraceRecallMemory,
    SelectiveWriteRecallMemory,
)
from neural_memory.contextbench_data import (
    ContextCandidate,
    ContextEpisode,
    load_contextbench_episodes,
)
from neural_memory.text_encoders import build_echo_inputs, pool_last_token, pool_mean
from prepare_contextbench_gemma_layers import build_payload, unique_texts
from prepare_contextbench_split_features import split_context
from train_gemma_lora_pilot import project_folds, sample_episode_sets, sample_pairs
from train_context_activation_cv import shuffle_within_projects, transform_features
from train_context_write_recall_cv import balanced_project_folds, hard_selection_metrics


def _record(task: str, content: str) -> dict[str, object]:
    return {
        "instance_id": task,
        "repo": "example/project",
        "problem_statement": f"fix {task}",
        "gold_context": json.dumps(
            [{"file": f"{task}.py", "start_line": 1, "end_line": 2, "content": content}]
        ),
    }


def test_other_tasks_in_same_project_become_hard_negatives(tmp_path: Path) -> None:
    path = tmp_path / "contextbench.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in (_record("a", "alpha"), _record("b", "beta")))
    )
    episodes = load_contextbench_episodes(path)
    assert len(episodes) == 2
    assert all(sum(candidate.active for candidate in episode.candidates) == 1 for episode in episodes)
    assert all(len(episode.candidates) == 2 for episode in episodes)


def test_memory_observes_then_activates_compact_traces() -> None:
    model = MultiTraceRecallMemory(feature_dim=8, memory_dim=4)
    candidates = torch.randn(2, 3, 8)
    queries = torch.randn(2, 8)
    state = model.observe(candidates)
    assert state.traces.shape == (2, 3, 4)
    assert model.activate(state, queries).shape == (2, 3)
    assert model.alignment_logits(state, queries).shape == (2, 3)


def test_dual_memory_observes_normalized_traces() -> None:
    model = DualEncoderRecallMemory(feature_dim=8, memory_dim=4)
    candidates = torch.randn(2, 3, 8)
    queries = torch.randn(2, 8)
    state = model.observe(candidates)
    assert state.traces.shape == (2, 3, 4)
    assert torch.allclose(state.traces.norm(dim=-1), torch.ones(2, 3), atol=1e-5)
    assert model.activate(state, queries).shape == (2, 3)


def test_selective_memory_writes_only_to_capacity_from_early_context() -> None:
    model = SelectiveWriteRecallMemory(feature_dim=8, memory_dim=4, keep_ratio=0.5)
    candidates = torch.randn(2, 4, 8)
    write_context = torch.randn(2, 8)
    recall_context = torch.randn(2, 8)
    masks = torch.tensor([[True, True, True, True], [True, True, False, False]])
    logits, state = model(candidates, write_context, recall_context, masks)
    assert logits.shape == (2, 4)
    assert state.selected.sum(dim=1).tolist() == [2, 1]
    assert not bool(state.selected[~masks].any())
    logits[masks].sum().backward()
    assert model.write_gate[-1].weight.grad is not None
    assert bool(torch.isfinite(model.write_gate[-1].weight.grad).all())


def test_context_split_uses_title_then_body_with_short_fallback() -> None:
    assert split_context("title\nbody\nmore") == ("title", "body\nmore")
    assert split_context("one two three four") == ("one two", "three four")


def test_write_recall_helpers_balance_projects_and_score_hard_capacity() -> None:
    folds = balanced_project_folds(torch.tensor([10, 10, 20, 30, 30, 30]), 2)
    assert set(folds[0]).isdisjoint(folds[1])
    selected = torch.tensor([[True, False, True, False]])
    targets = torch.tensor([[True, False, False, True]])
    masks = torch.ones_like(targets)
    metrics = hard_selection_metrics(selected, targets, masks)
    assert metrics.balanced_accuracy == 0.5
    assert metrics.precision == 0.5


def test_query_shuffle_stays_inside_project_groups() -> None:
    queries = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    projects = torch.tensor([10, 10, 20, 20])
    shuffled = shuffle_within_projects(queries, projects)
    assert shuffled.tolist() == [[2.0], [1.0], [4.0], [3.0]]


def test_feature_transform_fits_only_training_rows() -> None:
    candidates = torch.tensor([[[1.0, 0.0]], [[100.0, 0.0]]])
    queries = torch.tensor([[1.0, 0.0], [100.0, 0.0]])
    masks = torch.ones(2, 1, dtype=torch.bool)
    transformed, transformed_queries = transform_features(
        candidates,
        queries,
        masks,
        torch.tensor([True, False]),
        mode="center-l2",
    )
    assert transformed[0].tolist() == [[0.0, 0.0]]
    assert transformed_queries[0].tolist() == [0.0, 0.0]
    assert transformed[1].tolist() == [[1.0, 0.0]]


def test_last_token_pooling_ignores_right_padding() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0], [99.0, 99.0]],
            [[3.0, 30.0], [4.0, 40.0], [5.0, 50.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert pool_last_token(hidden, mask).tolist() == [[2.0, 20.0], [5.0, 50.0]]


def test_last_token_pooling_supports_left_padding() -> None:
    hidden = torch.tensor([[[99.0], [2.0], [3.0]]])
    mask = torch.tensor([[0, 1, 1]])
    assert pool_last_token(hidden, mask).tolist() == [[3.0]]


def test_last_token_pooling_rejects_empty_sequences() -> None:
    hidden = torch.zeros(1, 2, 3)
    mask = torch.zeros(1, 2, dtype=torch.long)
    try:
        pool_last_token(hidden, mask)
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("empty sequences must be rejected")


def test_mean_pooling_ignores_padding() -> None:
    hidden = torch.tensor([[[1.0, 10.0], [3.0, 30.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert pool_mean(hidden, mask).tolist() == [[2.0, 20.0]]


def test_echo_inputs_select_only_second_copy() -> None:
    class FakeTokenizer:
        bos_token_id = 2
        pad_token_id = 0

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert text == "\n\n" and not add_special_tokens
            return [9]

        def __call__(self, texts: list[str], **_: object) -> dict[str, list[list[int]]]:
            assert texts == ["short", "long"]
            return {"input_ids": [[3], [4, 5]]}

    encoded, selected = build_echo_inputs(
        FakeTokenizer(), ("short", "long"), max_length=8, device="cpu"
    )
    assert encoded["input_ids"].tolist() == [[2, 3, 9, 3, 0, 0], [2, 4, 5, 9, 4, 5]]
    assert encoded["attention_mask"].tolist() == [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]
    assert selected.tolist() == [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1]]


def test_layer_payload_preserves_episode_labels() -> None:
    rows = [
        ContextEpisode(
            task_id="a",
            project="repo/a",
            query_context="query",
            candidates=(
                ContextCandidate(context="positive", active=True),
                ContextCandidate(context="negative", active=False),
            ),
        )
    ]
    cache = {text: torch.ones(3) for text in unique_texts(rows)}
    payload = build_payload(rows, cache, model_name="gemma", pooling="mean", layer=6)
    assert payload["all_candidates"].shape == (1, 2, 3)
    assert payload["all_targets"].tolist() == [[True, False]]
    assert payload["layer"] == 6


def test_combined_features_require_aligned_labels() -> None:
    base = {
        "all_candidates": torch.tensor([[[1.0, 0.0]]]),
        "all_queries": torch.tensor([[1.0, 0.0]]),
        "all_targets": torch.tensor([[True]]),
        "all_masks": torch.tensor([[True]]),
        "all_project_ids": torch.tensor([1]),
        "encoder_model": "left",
    }
    other = {**base, "encoder_model": "right"}
    combined = combine_payloads(base, other)
    assert combined["all_candidates"].shape == (1, 1, 4)
    assert torch.allclose(combined["all_candidates"].norm(dim=-1), torch.ones(1, 1))
    bad = {**other, "all_targets": torch.tensor([[False]])}
    try:
        combine_payloads(base, bad)
    except ValueError as error:
        assert "same episodes" in str(error)
    else:
        raise AssertionError("misaligned feature artifacts must be rejected")


def test_lora_pilot_project_folds_do_not_overlap() -> None:
    rows = [
        ContextEpisode(
            project=f"repo/{index}",
            task_id=str(index),
            query_context=f"query {index}",
            candidates=(ContextCandidate(context=f"candidate {index}", active=True),),
        )
        for index in range(4)
    ]
    folds = project_folds(rows, 2)
    assert set(folds[0]).isdisjoint(folds[1])
    queries, candidates, targets = sample_pairs(rows, __import__("random").Random(7), 3)
    assert len(queries) == len(candidates) == len(targets) == 3
    queries, groups, targets, masks = sample_episode_sets(
        rows, __import__("random").Random(7), 3
    )
    assert len(queries) == len(groups) == 3
    assert targets.shape == masks.shape == (3, 1)
    assert bool(masks.all())
