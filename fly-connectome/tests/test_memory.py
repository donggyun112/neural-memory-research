import numpy as np
import pytest
import scipy.sparse as sp

from memory import kc_mbon_weights, mbon_drive, write, write_budgeted, write_competitive, write_magnitude_weighted, write_surprise_gated


def test_kc_mbon_weights_slices_the_right_submatrix():
    # 4 neurons total; KC = positions [0, 2], MBON = positions [1, 3]
    matrix = sp.csr_matrix(
        np.array(
            [
                [0, 5, 0, 6],
                [0, 0, 0, 0],
                [0, 7, 0, 8],
                [0, 0, 0, 0],
            ],
            dtype=float,
        )
    )
    weights = kc_mbon_weights(matrix, np.array([0, 2]), np.array([1, 3]))
    assert np.array_equal(weights, [[5, 6], [7, 8]])


def test_mbon_drive_is_the_weighted_sum_over_active_kc():
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    drive = mbon_drive(weights, np.array([1.0, 0.0]))
    assert list(drive) == [1.0, 2.0]


def test_write_depresses_only_active_kc_rows():
    weights = np.array([[10.0, 10.0], [10.0, 10.0]])
    write(weights, np.array([1.0, 0.0]), eta=0.5)
    assert list(weights[0]) == [5.0, 5.0]
    assert list(weights[1]) == [10.0, 10.0]


def test_recalling_the_same_code_twice_gives_a_lower_drive_the_second_time():
    weights = np.array([[10.0, 10.0], [10.0, 10.0]])
    code = np.array([1.0, 1.0])

    first = mbon_drive(weights, code).sum()
    write(weights, code, eta=0.3)
    second = mbon_drive(weights, code).sum()

    assert second < first


def test_write_budgeted_conserves_total_weight():
    weights = np.array([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]])
    total_before = weights.sum()
    write_budgeted(weights, np.array([1.0, 0.0, 0.0]), eta=0.5)
    assert weights.sum() == pytest.approx(total_before)
    # the depressed row is still lower than the untouched rows it borrowed from
    assert weights[0, 0] < weights[1, 0]


def test_write_competitive_splits_a_fixed_budget_evenly_across_equal_shares():
    two_active = np.array([[10.0], [10.0], [10.0]])
    write_competitive(two_active, np.array([1.0, 1.0, 0.0]), eta=0.4)
    # each of 2 equally-active KCs gets half the budget: depressed by 0.2
    assert two_active[0, 0] == pytest.approx(8.0)
    assert two_active[1, 0] == pytest.approx(8.0)
    assert two_active[2, 0] == pytest.approx(10.0)


def test_write_competitive_gives_each_kc_a_smaller_hit_as_more_compete():
    few_active = np.array([[10.0], [10.0]])
    write_competitive(few_active, np.array([1.0, 1.0]), eta=0.4)  # 2 active, share 0.5 each

    many_active = np.array([[10.0], [10.0], [10.0], [10.0]])
    write_competitive(many_active, np.array([1.0, 1.0, 1.0, 1.0]), eta=0.4)  # 4 active, share 0.25 each

    # more competitors sharing the same fixed budget -> smaller hit per KC
    assert many_active[0, 0] > few_active[0, 0]


def test_write_competitive_is_a_no_op_with_nothing_active():
    weights = np.array([[10.0], [10.0]])
    write_competitive(weights, np.array([0.0, 0.0]), eta=0.5)
    assert list(weights.flatten()) == [10.0, 10.0]


def test_write_magnitude_weighted_gives_the_strongest_row_the_full_eta():
    weights = np.array([[10.0], [10.0]])
    write_magnitude_weighted(weights, np.array([1.0, 0.5]), eta=0.4)
    assert weights[0, 0] == pytest.approx(6.0)  # relative=1.0 -> full eta
    assert weights[1, 0] == pytest.approx(8.0)  # relative=0.5 -> half eta


def test_write_magnitude_weighted_two_codes_sharing_active_rows_get_different_depression():
    # this is the phase 31 case: same active KC set, different magnitudes
    weights_a = np.array([[10.0], [10.0]])
    weights_b = np.array([[10.0], [10.0]])
    write_magnitude_weighted(weights_a, np.array([1.0, 1.0]), eta=0.5)  # equal magnitudes
    write_magnitude_weighted(weights_b, np.array([1.0, 0.1]), eta=0.5)  # very unequal
    assert not np.array_equal(weights_a, weights_b)


def test_write_magnitude_weighted_is_a_no_op_with_nothing_active():
    weights = np.array([[10.0], [10.0]])
    write_magnitude_weighted(weights, np.array([0.0, 0.0]), eta=0.5)
    assert list(weights.flatten()) == [10.0, 10.0]


def test_write_surprise_gated_gives_a_fresh_kc_the_full_eta():
    initial = np.array([[10.0, 10.0], [10.0, 10.0]])
    weights = initial.copy()
    write_surprise_gated(weights, initial, np.array([1.0, 0.0]), eta=0.5)
    assert weights[0, 0] == pytest.approx(5.0)  # fully fresh -> full 0.5 eta
    assert weights[1, 0] == 10.0  # untouched, not active


def test_write_surprise_gated_gives_a_habituated_kc_a_smaller_hit():
    initial = np.array([[10.0, 10.0]])
    weights = np.array([[2.0, 2.0]])  # already depressed to 20% of pristine
    write_surprise_gated(weights, initial, np.array([1.0]), eta=0.5)
    # freshness = 0.2, so effective eta = 0.5 * 0.2 = 0.1 -> 2.0 * 0.9 = 1.8
    assert weights[0, 0] == pytest.approx(1.8)


def test_write_surprise_gated_repeated_writes_show_diminishing_depression():
    initial = np.array([[10.0, 10.0]])
    weights = initial.copy()
    code = np.array([1.0])

    write_surprise_gated(weights, initial, code, eta=0.5)
    first_drop = initial[0, 0] - weights[0, 0]
    before_second = weights[0, 0]
    write_surprise_gated(weights, initial, code, eta=0.5)
    second_drop = before_second - weights[0, 0]

    assert second_drop < first_drop  # habituation: each repeat changes it less than the last
