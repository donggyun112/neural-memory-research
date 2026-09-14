"""A real transduction pathway for sound, not text: synthesize simple
tones, measure their spectral energy in the frequency band Johnston's
organ groups A/B are known to respond to, and inject that energy into the
real JO-A/B neurons -- the sound-tuned groups fetch_johnstons_organ.py
identified (13 neurons total, phase 9's flagged limitation).

Phase 29 of this experiment line (see README.md). Per Kamikouchi et al.
2009, JO-A/B respond to near-field sound/antennal vibration, courtship-song
range roughly 200-350 Hz; JO-C/E respond to static deflection (gravity/wind)
instead, which a pure tone doesn't produce, so this deliberately leaves them
at zero rather than inventing a mapping for them. No random projection
anywhere -- band energy overlapping a real physiological tuning range IS
the activation, unlike text's random PN projection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fetch_johnstons_organ import jo_group
from kc_response import propagate

SAMPLE_RATE = 10_000
DURATION = 0.5
COURTSHIP_BAND = (200.0, 350.0)  # Hz, Kamikouchi et al. 2009


def synthesize_tone(frequency: float, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)
    if noise > 0:
        signal = signal + noise * np.random.default_rng(seed).standard_normal(len(t))
    return signal


def synthesize_white_noise(seed: int = 0) -> np.ndarray:
    n = int(SAMPLE_RATE * DURATION)
    return np.random.default_rng(seed).standard_normal(n)


def band_energy(waveform: np.ndarray, band: tuple[float, float] = COURTSHIP_BAND) -> float:
    """Fraction of total spectral energy falling inside `band`."""
    spectrum = np.abs(np.fft.rfft(waveform)) ** 2
    freqs = np.fft.rfftfreq(len(waveform), d=1.0 / SAMPLE_RATE)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    total = spectrum.sum()
    return float(spectrum[in_band].sum() / total) if total > 0 else 0.0


def build_stimulus(waveform: np.ndarray, neurons_df: pd.DataFrame) -> np.ndarray:
    groups = neurons_df["type"].map(jo_group)
    stimulus = np.zeros(len(neurons_df))
    ab_mask = groups.isin(["A", "B"]).to_numpy()
    stimulus[ab_mask] = band_energy(waveform)
    return stimulus


def filterbank_energies(waveform: np.ndarray, n_bands: int, low: float = 100.0, high: float = 5000.0) -> np.ndarray:
    """Energy fraction per log-spaced band -- NOT literature-verified
    per-neuron tuning (phase 30's honest caveat: real JO-A/B tuning curves
    are broad and overlapping, not sharp non-overlapping filters, and this
    goes beyond the literature-verified 200-350 Hz range to make
    discrimination possible at all). A necessary extension past phase 29's
    stricter single band, not a claim about real per-neuron frequency
    selectivity."""
    spectrum = np.abs(np.fft.rfft(waveform)) ** 2
    freqs = np.fft.rfftfreq(len(waveform), d=1.0 / SAMPLE_RATE)
    total = spectrum.sum()
    energies = np.zeros(n_bands)
    if total <= 0:
        return energies
    edges = np.geomspace(low, high, n_bands + 1)
    for i in range(n_bands):
        in_band = (freqs >= edges[i]) & (freqs < edges[i + 1])
        energies[i] = spectrum[in_band].sum() / total
    return energies


def build_stimulus_multichannel(waveform: np.ndarray, neurons_df: pd.DataFrame) -> np.ndarray:
    """Each real JO-A/B neuron gets a DIFFERENT frequency band's energy,
    instead of every neuron getting the same scalar (phase 29's
    build_stimulus) -- phase 30 found the single-scalar version makes every
    sound activate the identical downstream set, differing only in
    magnitude. This gives genuine multi-channel structure so different
    sounds can produce different patterns, at the cost of the band-to-
    neuron assignment being arbitrary (sorted by bodyId), not verified."""
    groups = neurons_df["type"].map(jo_group)
    ab_positions = np.flatnonzero(groups.isin(["A", "B"]).to_numpy())
    stimulus = np.zeros(len(neurons_df))
    stimulus[ab_positions] = filterbank_energies(waveform, n_bands=len(ab_positions))
    return stimulus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=Path(__file__).parent / "data_ammc", type=Path)
    args = parser.parse_args()

    neurons_df = pd.read_parquet(args.data_dir / "neurons.parquet")
    matrix = sp.load_npz(args.data_dir / "adjacency.npz")
    ab_mask = neurons_df["type"].map(jo_group).isin(["A", "B"]).to_numpy()
    other_mask = ~neurons_df["type"].map(jo_group).isin(["A", "B", "C", "E", "F"]).to_numpy()
    print(f"JO-A/B neurons (sound-tuned): {int(ab_mask.sum())}, downstream targets: {int(other_mask.sum())}\n")

    test_sounds = {
        "250Hz tone (courtship-song range)": synthesize_tone(250.0),
        "2000Hz tone (outside the band)": synthesize_tone(2000.0),
        "white noise": synthesize_white_noise(),
    }

    for name, waveform in test_sounds.items():
        stimulus = build_stimulus(waveform, neurons_df)
        drive = np.maximum(0.0, propagate(stimulus, matrix))
        print(f"{name}: band_energy={band_energy(waveform):.3f}, "
              f"downstream drive sum={drive[other_mask].sum():.4f}")


if __name__ == "__main__":
    main()
