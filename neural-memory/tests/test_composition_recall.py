from __future__ import annotations

import numpy as np
import pytest

from eval_composition_recall import delta_store, recall_at


def test_recall_counts_the_share_of_the_evidence_set_returned() -> None:
    assert recall_at(np.array([1, 5]), np.array([1, 2])) == pytest.approx(0.5)
    assert recall_at(np.array([1, 2]), np.array([1, 2])) == pytest.approx(1.0)
    assert recall_at(np.array([7]), np.array([1, 2])) == pytest.approx(0.0)


def test_the_store_returns_each_key_its_own_value() -> None:
    # Orthogonal keys are the case the delta rule solves exactly, so a read has
    # to come back as the value that was written there.
    keys = np.eye(3)
    values = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    state = delta_store(keys, values)
    for key, value in zip(keys, values):
        assert np.allclose(state @ key, value)


def test_a_later_write_on_the_same_key_replaces_the_earlier_value() -> None:
    keys = np.array([[1.0, 0.0], [1.0, 0.0]])
    values = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(delta_store(keys, values) @ keys[0], values[1])


def test_an_empty_store_returns_nothing() -> None:
    state = delta_store(np.zeros((0, 3)), np.zeros((0, 2)))
    assert state.shape == (2, 3)
    assert not state.any()
