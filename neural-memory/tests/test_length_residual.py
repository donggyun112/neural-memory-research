from __future__ import annotations

import pytest
import torch

from train_length_residual import build_variant, length_features


def test_length_features_rank_the_longest_candidate_last() -> None:
    lengths = torch.tensor([[10.0, 40.0, 20.0, 30.0]])
    features = length_features(lengths, log_mean=3.0, log_std=1.0)
    assert features.shape == (1, 4, 2)
    ranks = features[0, :, 1]
    assert ranks[1] == pytest.approx(1.0)
    assert ranks[0] == pytest.approx(0.0)
    # The standardized channel must stay monotone in raw length.
    assert torch.all(features[0, :, 0].argsort() == lengths[0].argsort())


def test_length_features_reject_a_degenerate_scale() -> None:
    with pytest.raises(ValueError):
        length_features(torch.ones(1, 4), log_mean=0.0, log_std=0.0)


def test_build_variant_keeps_recall_context_width_aligned() -> None:
    embeddings = torch.zeros(3, 4, 8)
    feedback = torch.zeros(3, 8)
    lengths = torch.zeros(3, 4, 2)
    for name, width in (("embedding", 8), ("length", 2), ("embedding_length", 10)):
        candidates, context = build_variant(name, embeddings, feedback, lengths)
        assert candidates.shape[-1] == width
        assert context.shape == (3, width)


def test_build_variant_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError):
        build_variant("nope", torch.zeros(1, 2, 3), torch.zeros(1, 3), torch.zeros(1, 2, 2))
