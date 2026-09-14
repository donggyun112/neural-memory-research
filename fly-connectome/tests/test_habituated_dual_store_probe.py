import numpy as np
import pytest

from habituated_dual_store_probe import dual_drive, habituated_dual_write


def test_fresh_kc_gets_full_eta_on_both_stores():
    initial = np.full((2, 2), 10.0)
    fast, slow = initial.copy(), initial.copy()
    habituated_dual_write(fast, slow, initial, np.array([1.0, 0.0]))
    assert fast[0, 0] == pytest.approx(5.0)  # FAST_ETA=0.5, freshness=1.0
    assert slow[0, 0] == pytest.approx(9.7)  # SLOW_ETA=0.03, freshness=1.0


def test_repeated_writes_show_diminishing_depression_in_both_stores():
    initial = np.full((1, 2), 10.0)
    fast, slow = initial.copy(), initial.copy()
    code = np.array([1.0])

    habituated_dual_write(fast, slow, initial, code)
    fast_drop_1 = initial[0, 0] - fast[0, 0]
    slow_drop_1 = initial[0, 0] - slow[0, 0]
    fast_before, slow_before = fast[0, 0], slow[0, 0]

    habituated_dual_write(fast, slow, initial, code)
    fast_drop_2 = fast_before - fast[0, 0]
    slow_drop_2 = slow_before - slow[0, 0]

    assert fast_drop_2 < fast_drop_1
    assert slow_drop_2 < slow_drop_1


def test_dual_drive_sums_both_stores():
    fast = np.array([[1.0, 2.0]])
    slow = np.array([[3.0, 4.0]])
    drive = dual_drive(fast, slow, np.array([1.0]))
    assert drive == pytest.approx(10.0)  # (1+2) + (3+4)
