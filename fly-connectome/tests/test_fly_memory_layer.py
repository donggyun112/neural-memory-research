import numpy as np
import pytest

from fly_memory_layer import FlyMemoryLayer


def test_constructs_from_a_plain_numpy_matrix_no_connectome_needed():
    weights = np.random.default_rng(0).random((10, 4))
    layer = FlyMemoryLayer(weights)
    assert layer.n_kc == 10
    assert layer.n_mbon == 4


def test_initial_state_gives_each_batch_item_its_own_copy_of_the_real_weights():
    weights = np.full((5, 2), 10.0)
    layer = FlyMemoryLayer(weights)
    state = layer.initial_state(batch_size=3)
    assert state.fast.shape == (3, 5, 2)
    assert state.slow.shape == (3, 5, 2)
    assert np.array_equal(state.fast[0], weights)
    assert np.array_equal(state.fast[1], weights)


def test_write_returns_a_new_state_and_does_not_mutate_the_old_one():
    weights = np.full((6, 3), 10.0)
    layer = FlyMemoryLayer(weights, fast_eta=0.5, slow_eta=0.03)
    state = layer.initial_state(batch_size=1)
    code = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

    next_state = layer.write(state, code)
    assert not np.array_equal(next_state.fast, state.fast)
    assert np.array_equal(state.fast, layer.initial_state(1).fast)  # old state untouched


def test_write_then_read_shows_recognition_on_exact_repeat():
    weights = np.full((6, 3), 10.0)
    layer = FlyMemoryLayer(weights, fast_eta=0.5, slow_eta=0.03)
    state = layer.initial_state(batch_size=1)
    code = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

    first = layer.read(state, code).sum()
    state = layer.write(state, code)
    second = layer.read(state, code).sum()

    assert second < first


def test_accepts_a_plain_1d_code_as_a_single_batch_item():
    weights = np.full((4, 2), 10.0)
    layer = FlyMemoryLayer(weights)
    state = layer.initial_state(batch_size=1)
    code_1d = np.array([1.0, 0.0, 0.0, 0.0])

    drive = layer.read(state, code_1d)
    assert drive.shape == (1, 2)


def test_batch_items_evolve_independently():
    weights = np.full((4, 2), 10.0)
    layer = FlyMemoryLayer(weights, fast_eta=0.5, slow_eta=0.03)
    state = layer.initial_state(batch_size=2)
    codes = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    state = layer.write(state, codes)
    # item 0 only touched KC 0, item 1 only touched KC 2
    assert state.fast[0, 0, 0] < 10.0
    assert state.fast[0, 2, 0] == 10.0
    assert state.fast[1, 0, 0] == 10.0
    assert state.fast[1, 2, 0] < 10.0


def test_batch_size_mismatch_is_rejected():
    weights = np.full((4, 2), 10.0)
    layer = FlyMemoryLayer(weights)
    state = layer.initial_state(batch_size=2)
    wrong_batch_code = np.zeros((3, 4))
    with pytest.raises(ValueError):
        layer.read(state, wrong_batch_code)


def test_wrong_kc_dimension_is_rejected():
    weights = np.full((4, 2), 10.0)
    layer = FlyMemoryLayer(weights)
    state = layer.initial_state(batch_size=1)
    wrong_dim_code = np.zeros((1, 999))
    with pytest.raises(ValueError):
        layer.read(state, wrong_dim_code)


def test_repeated_writes_habituate_instead_of_costing_the_same_every_time():
    weights = np.full((3, 2), 10.0)
    layer = FlyMemoryLayer(weights, fast_eta=0.5, slow_eta=0.03)
    state = layer.initial_state(batch_size=1)
    code = np.array([[1.0, 0.0, 0.0]])

    state = layer.write(state, code)
    first_drop = 10.0 - state.fast[0, 0, 0]
    before_second = state.fast[0, 0, 0]
    state = layer.write(state, code)
    second_drop = before_second - state.fast[0, 0, 0]

    assert second_drop < first_drop


def test_first_write_from_a_fresh_state_matches_the_flat_eta_rule():
    # freshness is 1.0 on a never-touched row, so the very first write must
    # be numerically identical to a plain (non-habituating) eta multiply.
    weights = np.full((2, 2), 10.0)
    layer = FlyMemoryLayer(weights, fast_eta=0.4, slow_eta=0.1)
    state = layer.initial_state(batch_size=1)
    state = layer.write(state, np.array([[1.0, 0.0]]))
    assert state.fast[0, 0, 0] == pytest.approx(6.0)  # 10 * (1 - 0.4)
    assert state.slow[0, 0, 0] == pytest.approx(9.0)  # 10 * (1 - 0.1)
