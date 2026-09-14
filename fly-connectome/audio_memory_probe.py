"""Write/recall memory on the sound circuit -- phase 29 only built the
input side (waveform -> real JO-A/B activation); this wires it through to
real write/recall, mirroring the text line (phase 3, 5) and the vision line
(phase 12, 14).

Phase 30 found phase 29's single-scalar `build_stimulus` makes every sound
activate the identical downstream set (only magnitude differs) -- fixed by
switching to `build_stimulus_multichannel` (a frequency filterbank across
the 13 real JO-A/B neurons). But checking further found a second issue:
only ~164 of 2631 downstream neurons are reachable AT ALL from this small
13-neuron population (a real anatomical bottleneck, not a modeling
artifact) -- so EVERY sound's *sparse-coded* (top-k, binary) active set is
still identical, just like phase 30, even though the underlying continuous
drive really does differ (raw-vector cosine similarity as low as 0.01
between very different sounds). The fix here: don't sparse-code at all --
this circuit is already naturally sparse from its own anatomy, and an
artificial top-k binarization on top of that throws away exactly the
magnitude pattern that carries the real content difference. Use the raw
(ReLU'd) drive as the write/recall code directly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from audio_to_jo import build_stimulus_multichannel, synthesize_tone, synthesize_white_noise
from fetch_johnstons_organ import jo_group
from kc_response import propagate
from memory import mbon_drive, write_magnitude_weighted

ETA = 0.3


def main() -> None:
    data_dir = Path(__file__).parent / "data_ammc"
    neurons_df = pd.read_parquet(data_dir / "neurons.parquet")
    matrix = sp.load_npz(data_dir / "adjacency.npz")

    groups = neurons_df["type"].map(jo_group)
    other_mask = ~groups.isin(["A", "B", "C", "E", "F"]).to_numpy()
    other_positions = np.flatnonzero(other_mask)
    n_other = len(other_positions)

    sounds = {
        "250Hz (in-band)": synthesize_tone(250.0),
        "2000Hz (out-of-band)": synthesize_tone(2000.0),
        "white noise": synthesize_white_noise(),
    }
    codes = {}
    for name, waveform in sounds.items():
        stimulus = build_stimulus_multichannel(waveform, neurons_df)
        drive = np.maximum(0.0, propagate(stimulus, matrix))
        codes[name] = drive[other_mask]
        print(f"{name}: {int((codes[name] > 0).sum())}/{n_other} downstream AMMC neurons reachable")

    print("\npairwise cosine similarity of the RAW (continuous) code -- not binary occupancy,")
    print("which is identical for every sound here (a real anatomical bottleneck, not a bug):")
    names = list(codes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va, vb = codes[a], codes[b]
            cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
            print(f"  {a} vs {b}: {cos:.4f}")

    weights = np.asarray(matrix[other_positions][:, other_positions].todense(), dtype=float)

    print("\nbaseline recall, no writes:")
    for name, code in codes.items():
        print(f"  {name:22s}: {float(mbon_drive(weights, code).sum()):10.4f}")

    target = "250Hz (in-band)"
    before = float(mbon_drive(weights, codes[target]).sum())
    write_magnitude_weighted(weights, codes[target], eta=ETA)
    after = float(mbon_drive(weights, codes[target]).sum())
    print(f"\n{target}, exact repeat: {before:.4f} -> {after:.4f}  ({100 * (after - before) / before:+.1f}%)")

    for name, code in codes.items():
        if name == target:
            continue
        drive_now = float(mbon_drive(weights, code).sum())
        baseline = float(mbon_drive(np.asarray(matrix[other_positions][:, other_positions].todense(), dtype=float), code).sum())
        print(f"{name}: baseline={baseline:.4f} -> after writing {target!r}={drive_now:.4f} "
              f"({100 * (drive_now - baseline) / baseline:+.1f}%)")


if __name__ == "__main__":
    main()
