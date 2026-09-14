import numpy as np
import pandas as pd

from odor_embedder import find_odor_match, glomerulus_pn_mask, grounded_activation


def test_find_odor_match_is_case_insensitive_and_word_bounded():
    match = find_odor_match("The soil smelled of Geosmin after the storm.")
    assert match is not None
    keyword, glomerulus, citation = match
    assert keyword == "geosmin"
    assert glomerulus == "DA2"


def test_find_odor_match_does_not_match_substrings_inside_other_words():
    # "ethanol" is a literal substring of "methanol" -- must not match without a word boundary
    assert find_odor_match("The spill was methanol, not something safer.") is None


def test_find_odor_match_returns_none_for_unrelated_text():
    assert find_odor_match("The fire alarm rang loudly in the empty hallway.") is None


def test_find_odor_match_finds_ethyl_butyrate():
    match = find_odor_match("The cocktail smelled faintly of ethyl butyrate.")
    assert match is not None
    keyword, glomerulus, _ = match
    assert keyword == "ethyl butyrate"
    assert glomerulus == "DM2"


def test_glomerulus_pn_mask_matches_only_the_named_glomerulus():
    neurons_df = pd.DataFrame(
        {"bodyId": [1, 2, 3, 4], "type": ["DA2_lPN", "DM1_lPN", "DA2_vPN", "MBON14"]}
    )
    mask = glomerulus_pn_mask(np.array([1, 2, 3]), neurons_df, "DA2")
    assert list(mask) == [True, False, True]


def test_grounded_activation_sets_only_matched_glomerulus_pns():
    neurons_df = pd.DataFrame({"bodyId": [1, 2, 3], "type": ["DA2_lPN", "DM1_lPN", "DA2_vPN"]})
    activation = grounded_activation(np.array([1, 2, 3]), neurons_df, "DA2", magnitude=2.0)
    assert list(activation) == [2.0, 0.0, 2.0]
