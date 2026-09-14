import numpy as np
import pytest

from recognition_under_load_probe import auc, depression_scores, overlap_scores


def test_auc_is_one_when_written_always_scores_higher():
    assert auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == pytest.approx(1.0)


def test_auc_is_half_when_the_groups_are_identical():
    # A mechanism that separates nothing must read 0.5, not an accidental 1.0:
    # every comparison is a tie and ties count as half.
    assert auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(0.5)


def test_auc_rejects_an_empty_group():
    with pytest.raises(ValueError):
        auc(np.array([1.0]), np.array([]))


def _codes(count: int = 4, kc: int = 6) -> list[np.ndarray]:
    codes = []
    for index in range(count):
        code = np.zeros(kc)
        # Disjoint active sets, so a write on one cannot touch another.
        code[index] = 1.0
        codes.append(code)
    return codes


def test_written_items_score_above_unwritten_ones():
    codes = _codes()
    initial = np.ones((6, 3))
    for mode in ("unbudgeted", "saturating", "dual", "dual_tuned", "habituation", "habituated_dual"):
        scores = depression_scores(initial, codes, [0, 1], mode)
        assert scores[0] > scores[2], mode
        assert scores[1] > scores[3], mode


def test_unwritten_items_are_untouched_when_codes_do_not_overlap():
    scores = depression_scores(np.ones((6, 3)), _codes(), [0], "unbudgeted")
    assert scores[1] == pytest.approx(0.0)


def test_overlap_control_recognises_exactly_what_it_stored():
    scores = overlap_scores(_codes(), [0, 1])
    assert scores[0] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(0.0)
