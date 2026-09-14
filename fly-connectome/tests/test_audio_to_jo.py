import numpy as np
import pandas as pd

from audio_to_jo import (
    band_energy,
    build_stimulus,
    build_stimulus_multichannel,
    filterbank_energies,
    synthesize_tone,
    synthesize_white_noise,
)


def test_band_energy_is_high_for_a_tone_inside_the_band():
    tone = synthesize_tone(275.0)  # centered in the 200-350 Hz band
    assert band_energy(tone) > 0.9


def test_band_energy_is_low_for_a_tone_outside_the_band():
    tone = synthesize_tone(2000.0)
    assert band_energy(tone) < 0.1


def test_band_energy_is_moderate_for_white_noise():
    noise = synthesize_white_noise()
    # broadband noise spreads energy across the whole spectrum, so only a
    # small, non-zero fraction should land in a 150 Hz-wide band
    assert 0.0 < band_energy(noise) < 0.3


def test_build_stimulus_only_activates_jo_ab_neurons():
    neurons_df = pd.DataFrame({"bodyId": [1, 2, 3], "type": ["JO-A1", "JO-C1", "BM"]})
    stimulus = build_stimulus(synthesize_tone(275.0), neurons_df)
    assert stimulus[0] > 0  # JO-A
    assert stimulus[1] == 0  # JO-C, not sound-tuned
    assert stimulus[2] == 0  # not JO at all


def test_filterbank_energies_puts_a_low_tone_in_a_low_band():
    energies = filterbank_energies(synthesize_tone(150.0), n_bands=5, low=100.0, high=5000.0)
    assert np.argmax(energies) == 0


def test_filterbank_energies_puts_a_high_tone_in_a_high_band():
    energies = filterbank_energies(synthesize_tone(4000.0), n_bands=5, low=100.0, high=5000.0)
    assert np.argmax(energies) == 4


def test_build_stimulus_multichannel_gives_different_sounds_different_patterns():
    neurons_df = pd.DataFrame({"bodyId": list(range(4)), "type": ["JO-A1", "JO-A4", "JO-B1_a", "BM"]})
    low_tone = build_stimulus_multichannel(synthesize_tone(150.0), neurons_df)
    high_tone = build_stimulus_multichannel(synthesize_tone(4000.0), neurons_df)
    # different sounds should NOT be simple scalar multiples of each other
    # (phase 29's single-scalar design was exactly that, and phase 30 found
    # it made every sound indistinguishable downstream)
    assert not np.allclose(low_tone[:3] / (low_tone[:3].sum() or 1), high_tone[:3] / (high_tone[:3].sum() or 1))
    assert high_tone[3] == 0  # not JO at all, never touched
