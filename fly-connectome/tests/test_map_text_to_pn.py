import numpy as np
import pandas as pd

from map_text_to_pn import load_pn_positions, project_to_pn, random_projection, split_sentences


def test_split_sentences_handles_punctuation_and_newlines():
    text = "First one. Second one!\nThird one? trailing"
    assert split_sentences(text) == ["First one.", "Second one!", "Third one?", "trailing"]


def test_split_sentences_ignores_blank_input():
    assert split_sentences("   \n  ") == []


def test_random_projection_is_deterministic_given_a_seed():
    a = random_projection(8, 4, seed=1)
    b = random_projection(8, 4, seed=1)
    c = random_projection(8, 4, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (8, 4)


def test_project_to_pn_is_never_negative():
    embeddings = np.array([[1.0, -1.0], [-5.0, -5.0]])
    projection = random_projection(2, 6, seed=0)
    activations = project_to_pn(embeddings, projection)
    assert activations.shape == (2, 6)
    assert (activations >= 0).all()


def test_load_pn_positions_matches_matrix_row_order_not_sorted_bodyid(tmp_path):
    # bodyId order is deliberately NOT sorted, to catch any code that
    # silently re-sorts and breaks alignment with the adjacency matrix.
    neurons_df = pd.DataFrame(
        {"bodyId": [500, 100, 300, 200], "type": ["il3PN", "KCg-m", "VA1v_PN", "MBON14"]}
    )
    path = tmp_path / "neurons.parquet"
    neurons_df.to_parquet(path)

    positions, pn_ids = load_pn_positions(path)
    assert list(positions) == [0, 2]
    assert list(pn_ids) == [500, 300]
