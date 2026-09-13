from __future__ import annotations

import torch

from train_revisit_write_recall import (
    add_causal_write_features,
    evaluate,
    train_model,
)


def test_revisit_model_trains_write_then_recall_without_target_at_inference() -> None:
    torch.manual_seed(3)
    candidates = torch.randn(16, 8, 6)
    feedback = torch.randn(16, 6)
    targets = torch.arange(16) % 8
    model = train_model(
        candidates,
        feedback,
        targets,
        keep_ratio=0.75,
        write_steps=2,
        recall_steps=2,
        batch_size=8,
        memory_dim=4,
        learning_rate=1e-3,
        seed=7,
        shuffle_write_teacher=False,
        device=torch.device("cpu"),
    )
    result = evaluate(model, candidates, feedback, targets, device=torch.device("cpu"))
    assert 0.0 <= result.retained_target_rate <= 1.0
    assert 0.0 <= result.write_balanced_accuracy <= 1.0
    assert 0.0 <= result.top1 <= result.top2 <= 1.0


def test_causal_write_features_do_not_look_at_later_slots() -> None:
    torch.manual_seed(5)
    candidates = torch.randn(3, 5, 4)
    feedback = torch.randn(3, 4)
    changed = candidates.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 100
    original_features, original_feedback = add_causal_write_features(candidates, feedback)
    changed_features, changed_feedback = add_causal_write_features(changed, feedback)
    torch.testing.assert_close(original_features[:, :3], changed_features[:, :3])
    torch.testing.assert_close(original_feedback, changed_feedback)


def test_listwise_joint_training_smoke() -> None:
    torch.manual_seed(9)
    candidates = torch.randn(16, 8, 6)
    feedback = torch.randn(16, 6)
    targets = torch.arange(16) % 8
    model = train_model(
        candidates,
        feedback,
        targets,
        keep_ratio=0.25,
        write_steps=2,
        recall_steps=2,
        batch_size=8,
        memory_dim=4,
        learning_rate=1e-3,
        seed=11,
        shuffle_write_teacher=False,
        write_objective="listwise",
        causal_features=True,
        joint_steps=2,
        device=torch.device("cpu"),
    )
    result = evaluate(
        model,
        candidates,
        feedback,
        targets,
        causal_features=True,
        device=torch.device("cpu"),
    )
    assert 0.0 <= result.retained_target_rate <= 1.0
