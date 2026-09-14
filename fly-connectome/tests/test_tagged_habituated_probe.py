import numpy as np

import tagged_habituated_probe as m


def test_write_event_gives_a_fresh_kc_the_full_habituated_eta():
    initial = np.full((3, 2), 10.0)
    fast = initial.copy()
    tag = np.zeros(3)
    m.write_event(fast, tag, initial, np.array([1.0, 0.0, 0.0]))
    assert fast[0, 0] == 5.0  # freshness=1.0 -> full FAST_ETA=0.5
    assert list(tag) == [1.0, 0.0, 0.0]


def test_write_event_gives_a_habituated_kc_a_smaller_hit():
    initial = np.full((1, 2), 10.0)
    fast = np.array([[2.0, 2.0]])  # already at 20% of pristine
    tag = np.zeros(1)
    m.write_event(fast, tag, initial, np.array([1.0]))
    # freshness=0.2 -> effective eta = 0.5*0.2 = 0.1 -> 2.0*0.9=1.8
    assert fast[0, 0] == 1.8


def test_capture_event_only_moves_slow_toward_zero_never_up():
    initial = np.full((3, 2), 10.0)
    slow = initial.copy()
    tag = np.array([1.0, 0.5, 0.0])
    m.capture_event(slow, tag, initial)
    assert slow[0, 0] < initial[0, 0]
    assert slow[1, 0] < initial[1, 0]
    assert slow[2, 0] == initial[2, 0]  # untagged, untouched


def test_recall_sums_fast_and_slow():
    fast = np.array([[1.0, 2.0]])
    slow = np.array([[3.0, 4.0]])
    assert m.recall(fast, slow, np.array([1.0])) == 10.0
