import numpy as np

from stimulus_points import jaccard, stimulus_scores


def test_jaccard_of_identical_sets_is_one():
    assert jaccard(frozenset({1, 2, 3}), frozenset({1, 2, 3})) == 1.0


def test_jaccard_of_disjoint_sets_is_zero():
    assert jaccard(frozenset({1, 2}), frozenset({3, 4})) == 0.0


def test_jaccard_of_two_empty_sets_is_one():
    assert jaccard(frozenset(), frozenset()) == 1.0


def test_stimulus_scores_ranks_the_odd_one_out_highest():
    # Two near-identical codes, one completely disjoint from both.
    codes = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    scores = stimulus_scores(codes)
    assert np.argmax(scores) == 2
    assert scores[0] < scores[2] and scores[1] < scores[2]


def test_stimulus_scores_is_zero_with_fewer_than_two_sentences():
    assert list(stimulus_scores(np.array([[1.0, 0.0]]))) == [0.0]
