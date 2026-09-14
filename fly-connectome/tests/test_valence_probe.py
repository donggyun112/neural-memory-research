import numpy as np

from valence_probe import valence_score, write_signed


def _setup():
    # 2 KC, 4 MBON (2 positive, 2 negative)
    weights = np.full((2, 4), 10.0)
    positive_mask = np.array([True, True, False, False])
    code = np.array([1.0, 0.0])
    return weights, positive_mask, code


def test_positive_reinforcement_depresses_only_the_negative_columns():
    weights, positive_mask, code = _setup()
    write_signed(weights, code, positive_mask, sign=+1)
    assert weights[0, 0] == 10.0 and weights[0, 1] == 10.0  # positive cols untouched
    assert weights[0, 2] < 10.0 and weights[0, 3] < 10.0  # negative cols depressed
    assert weights[1, 0] == 10.0  # inactive KC row untouched entirely


def test_negative_reinforcement_depresses_only_the_positive_columns():
    weights, positive_mask, code = _setup()
    write_signed(weights, code, positive_mask, sign=-1)
    assert weights[0, 0] < 10.0 and weights[0, 1] < 10.0
    assert weights[0, 2] == 10.0 and weights[0, 3] == 10.0


def test_valence_score_is_positive_drive_minus_negative_drive():
    weights = np.array([[1.0, 2.0, 3.0, 4.0]])
    positive_mask = np.array([True, True, False, False])
    code = np.array([1.0])
    assert valence_score(weights, code, positive_mask) == (1.0 + 2.0) - (3.0 + 4.0)


def test_positive_reinforcement_shifts_the_readout_toward_positive():
    weights, positive_mask, code = _setup()
    before = valence_score(weights, code, positive_mask)
    write_signed(weights, code, positive_mask, sign=+1)
    after = valence_score(weights, code, positive_mask)
    assert after > before  # negative channel depressed -> difference grows positive


def test_negative_reinforcement_shifts_the_readout_toward_negative():
    weights, positive_mask, code = _setup()
    before = valence_score(weights, code, positive_mask)
    write_signed(weights, code, positive_mask, sign=-1)
    after = valence_score(weights, code, positive_mask)
    assert after < before
