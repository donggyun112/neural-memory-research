from fetch_optic_lobe import photoreceptor_class


def test_photoreceptor_class_matches_known_real_types():
    assert photoreceptor_class("R1-R6") == "R1-R6"
    assert photoreceptor_class("R7y") == "R7y"
    assert photoreceptor_class("R8p") == "R8p"
    assert photoreceptor_class("R7d") == "R7d"
    assert photoreceptor_class("R7_unclear") == "R7"
    assert photoreceptor_class("R7R8_unclear") == "R7R8"


def test_photoreceptor_class_falls_back_to_other():
    assert photoreceptor_class("L1") == "other"
    assert photoreceptor_class(None) == "other"
