import numpy as np
import scipy.sparse as sp

from kc_response import propagate, random_rewire, scatter, sparse_code


def test_scatter_places_values_at_given_positions_and_zeros_elsewhere():
    full = scatter(np.array([1.0, 2.0]), np.array([3, 0]), n_total=5)
    assert list(full) == [2.0, 0.0, 0.0, 1.0, 0.0]


def test_propagate_sums_weighted_incoming_drive():
    # neuron 0 -> neuron 1 (weight 3), neuron 0 -> neuron 2 (weight 1)
    matrix = sp.csr_matrix(np.array([[0, 3, 1], [0, 0, 0], [0, 0, 0]], dtype=float))
    stimulus = np.array([2.0, 0.0, 0.0])
    drive = propagate(stimulus, matrix)
    assert list(drive) == [0.0, 6.0, 2.0]


def test_sparse_code_keeps_only_the_top_fraction():
    drive = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
    coded = sparse_code(drive, keep_ratio=0.4)  # top 2 of 5
    assert (coded > 0).sum() == 2
    assert set(coded[coded > 0]) == {5.0, 4.0}


def test_sparse_code_keep_ratio_one_is_a_no_op():
    drive = np.array([1.0, 0.0, 3.0])
    assert list(sparse_code(drive, keep_ratio=1.0)) == list(drive)


def test_random_rewire_preserves_edge_count_and_weight_multiset_but_not_targets():
    matrix = sp.csr_matrix(np.array([[0, 3, 1], [0, 0, 2], [0, 0, 0]], dtype=float))
    rewired = random_rewire(matrix, seed=0)
    assert rewired.nnz == matrix.nnz
    assert sorted(rewired.data) == sorted(matrix.data)
    assert rewired.shape == matrix.shape


def test_random_rewire_is_deterministic_given_a_seed():
    matrix = sp.random(20, 20, density=0.3, format="csr", random_state=1)
    a = random_rewire(matrix, seed=5)
    b = random_rewire(matrix, seed=5)
    assert (a != b).nnz == 0
