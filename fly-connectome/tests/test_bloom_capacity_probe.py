import numpy as np
import pytest

from bloom_capacity_probe import predicted_auc, saturating_scores, shuffled_codes


def test_nothing_written_means_perfect_separation():
    assert predicted_auc(100, 2000, 0) == pytest.approx(1.0)


def test_enough_writes_drive_the_prediction_to_chance():
    # Every cell ends up marked, so an unwritten item is indistinguishable.
    assert predicted_auc(100, 2000, 100_000) == pytest.approx(0.5)


def test_the_prediction_falls_as_the_load_grows():
    values = [predicted_auc(100, 2000, n) for n in (1, 10, 100, 1000)]
    assert values == sorted(values, reverse=True)


def test_active_cells_must_fit_inside_the_population():
    with pytest.raises(ValueError):
        predicted_auc(3000, 2000, 10)


def _codes() -> np.ndarray:
    codes = np.zeros((3, 8))
    codes[0, [0, 1]] = 1.0
    codes[1, [2, 3]] = 1.0
    codes[2, [4, 5]] = 1.0
    return codes


def test_a_written_document_reads_as_fully_depressed():
    scores = saturating_scores(np.ones((8, 2)), _codes(), [0], eta=1.0, binary=True)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_counting_and_binary_differ_only_on_repeated_writes():
    initial = np.ones((8, 2))
    codes = _codes()
    once = saturating_scores(initial, codes, [0], eta=0.5, binary=False)
    twice = saturating_scores(initial, codes, [0, 0], eta=0.5, binary=False)
    clamped = saturating_scores(initial, codes, [0, 0], eta=0.5, binary=True)
    assert twice[0] > once[0]
    assert clamped[0] == pytest.approx(once[0])


def test_shuffling_keeps_the_count_of_active_cells():
    codes = _codes()
    shuffled = shuffled_codes(codes, np.random.default_rng(0))
    assert np.array_equal((shuffled > 0).sum(axis=1), (codes > 0).sum(axis=1))
