import numpy as np

from tagging_rescue_probe import episode_recall


def _codes():
    # 3 disjoint KC groups: target uses KC {0,1}, filler uses {2,3}, event uses {4,5}
    target = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    filler = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    event = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    return target, filler, event


def test_ceiling_shows_the_strongest_single_shot_depression():
    weights = np.full((6, 2), 10.0)
    target, filler, event = _codes()
    shared = dict(weights_template=weights, target_code=target, filler_codes=[filler], event_code=event,
                   weak_eta=0.05, strong_eta=0.3, tag_decay=0.7)

    ceiling = episode_recall(**shared, capture_rate=0.0, mode="ceiling")
    control = episode_recall(**shared, capture_rate=0.0, mode="control")
    assert ceiling < control  # writing the target strong depresses it more than writing it weak


def test_rescue_moves_recall_toward_the_ceiling_not_past_it():
    weights = np.full((6, 2), 10.0)
    target, filler, event = _codes()
    shared = dict(weights_template=weights, target_code=target, filler_codes=[filler], event_code=event,
                   weak_eta=0.05, strong_eta=0.3, tag_decay=0.7)

    control = episode_recall(**shared, capture_rate=0.0, mode="control")
    rescued = episode_recall(**shared, capture_rate=1.0, mode="rescued")
    ceiling = episode_recall(**shared, capture_rate=0.0, mode="ceiling")

    assert ceiling <= rescued < control  # rescued is between control and ceiling


def test_rescue_is_content_free_even_when_event_shares_no_kcs_with_target():
    # event_code (KCs {4,5}) never overlaps target_code (KCs {0,1}) -- rescue
    # must still happen, since capture depends only on the tag, not overlap.
    weights = np.full((6, 2), 10.0)
    target, filler, event = _codes()
    assert not (set(np.flatnonzero(target)) & set(np.flatnonzero(event)))

    shared = dict(weights_template=weights, target_code=target, filler_codes=[filler], event_code=event,
                   weak_eta=0.05, strong_eta=0.3, tag_decay=0.7)
    control = episode_recall(**shared, capture_rate=0.0, mode="control")
    rescued = episode_recall(**shared, capture_rate=1.0, mode="rescued")
    assert rescued < control


def test_longer_gap_gives_less_rescue_because_the_tag_has_decayed():
    weights = np.full((6, 2), 10.0)
    target, filler, event = _codes()

    def rescued_effect(n_fillers):
        shared = dict(weights_template=weights, target_code=target, filler_codes=[filler] * n_fillers,
                       event_code=event, weak_eta=0.05, strong_eta=0.3, tag_decay=0.5)
        control = episode_recall(**shared, capture_rate=0.0, mode="control")
        rescued = episode_recall(**shared, capture_rate=1.0, mode="rescued")
        return control - rescued  # how much extra depression the rescue added

    short_gap = rescued_effect(0)
    long_gap = rescued_effect(6)
    assert short_gap > long_gap
