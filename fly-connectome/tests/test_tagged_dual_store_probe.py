import numpy as np

import tagged_dual_store_probe as m


def test_write_event_depresses_only_fast_and_tags_active_kcs():
    fast = np.full((4, 2), 10.0)
    tag = np.zeros(4)
    code = np.array([1.0, 0.0, 1.0, 0.0])

    m.write_event(fast, tag, code)

    assert fast[0, 0] < 10.0 and fast[2, 0] < 10.0
    assert fast[1, 0] == 10.0 and fast[3, 0] == 10.0
    assert list(tag) == [1.0, 0.0, 1.0, 0.0]


def test_write_event_decays_existing_tag_before_refreshing_touched_kcs():
    fast = np.full((3, 1), 10.0)
    tag = np.array([1.0, 1.0, 1.0])
    m.write_event(fast, tag, np.array([1.0, 0.0, 0.0]))
    assert tag[0] == 1.0  # refreshed by this event
    assert tag[1] == m.TAG_DECAY  # decayed, not touched
    assert tag[2] == m.TAG_DECAY


def test_capture_event_only_affects_slow_and_only_tagged_kcs():
    slow = np.full((3, 2), 10.0)
    tag = np.array([1.0, 0.0, 0.5])
    m.capture_event(slow, tag)
    assert slow[0, 0] < 10.0  # fully tagged -> fully captured (capture_rate=1)
    assert slow[1, 0] == 10.0  # untagged -> untouched
    assert 0.0 < slow[2, 0] < 10.0  # partially tagged -> partially captured


def test_capture_event_does_not_touch_fast_store():
    fast = np.full((3, 2), 10.0)
    slow = np.full((3, 2), 10.0)
    tag = np.array([1.0, 1.0, 1.0])
    m.capture_event(slow, tag)
    assert np.array_equal(fast, np.full((3, 2), 10.0))
