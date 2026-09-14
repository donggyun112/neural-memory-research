from fetch_johnstons_organ import jo_group


def test_jo_group_extracts_the_letter_after_jo_dash():
    assert jo_group("JO-A1") == "A"
    assert jo_group("JO-A4") == "A"
    assert jo_group("JO-B1_a") == "B"
    assert jo_group("JO-B-unclear") == "B"
    assert jo_group("JO-CA1") == "C"
    assert jo_group("JO-CL") == "C"
    assert jo_group("JO-ED1") == "E"
    assert jo_group("JO-EV6") == "E"
    assert jo_group("JO-FV") == "F"


def test_jo_group_falls_back_to_other():
    assert jo_group("JO-unclear") == "other"
    assert jo_group("BM") == "other"
    assert jo_group(None) == "other"
