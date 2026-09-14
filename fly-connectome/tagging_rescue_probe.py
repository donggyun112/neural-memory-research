"""RESEARCH.md measurement 7: write an item weakly, apply a strong
unrelated event after k intervening writes, measure rescue against k --
the "deferral window the corpora could not supply."

Phase 20 (see README.md). Ports the design already validated in
experiments/neural-memory/analyze_tagging_window.py (git log: "a fading tag
produces the rescue window the corpora could not show") onto our real
KC->MBON depression mechanism, rather than reinventing the measurement.

Mapping onto a depression-only system (no outer-product value store, so
"better memory" here means MORE depression, not closer cosine to a stored
value): a weak write banks its *unrealized* depression (strong_eta -
weak_eta) as a per-KC tag; the tag decays each subsequent write; a later
strong event, at ANY unrelated KCs, releases a capture pulse that applies
extra depression to whatever KCs are still tagged -- content-free, exactly
as the reference implementation notes ("the event contributes only
magnitude"). Recall = MBON drive for the target's own code; rescue lowers
it (more consolidated), and recovered_fraction compares that against a
control (same events, capture=0) and a ceiling (target written strong from
the start).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fetch_mushroom_body import cell_class
from generalization_probe import kc_code_for
from map_text_to_pn import load_pn_positions
from memory import kc_mbon_weights, mbon_drive

DOCUMENT_POOL = [
    "The fire alarm rang loudly in the empty hallway.",
    "A loud alarm blared through the deserted corridor.",
    "Sirens wailed through the abandoned building at night.",
    "A cup of coffee sat cold on the desk.",
    "Her mug of tea grew cold on the table.",
    "An untouched glass of water rested on the counter.",
    "Rain tapped steadily against the window all afternoon.",
    "Drizzle fell softly on the glass throughout the day.",
    "Snow drifted quietly past the frosted pane that evening.",
    "The dog chased a ball across the sunny park.",
    "A puppy ran after a toy in the green field.",
    "Children played fetch with their pet near the meadow.",
    "The train arrived at the station exactly on time.",
    "A car passed slowly down the quiet street.",
    "He finished his homework just before dinner.",
    "The lights flickered twice during the evening storm.",
    "They walked along the river as the sun set.",
    "The clock on the wall stopped working at noon.",
    "She left her umbrella behind at the cafe.",
    "The old radio crackled with static all morning.",
]


def episode_recall(
    weights_template: np.ndarray,
    target_code: np.ndarray,
    filler_codes: list[np.ndarray],
    event_code: np.ndarray,
    *,
    weak_eta: float,
    strong_eta: float,
    tag_decay: float,
    capture_rate: float,
    mode: str,
) -> float:
    """mode: 'rescued' (capture applied), 'control' (same events, no
    capture), 'ceiling' (target written strong from the start, no tag)."""
    weights = weights_template.copy()
    n_kc = weights.shape[0]
    tag = np.zeros(n_kc)

    target_eta = strong_eta if mode == "ceiling" else weak_eta
    active_target = target_code > 0
    weights[active_target, :] *= 1.0 - target_eta
    if mode != "ceiling":
        tag[active_target] += strong_eta - weak_eta  # unrealized depression, banked

    for filler in filler_codes:
        tag *= tag_decay
        active = filler > 0
        weights[active, :] *= 1.0 - strong_eta

    tag *= tag_decay
    active_event = event_code > 0
    weights[active_event, :] *= 1.0 - strong_eta
    if mode == "rescued":
        # Content-free: applies to whatever is still tagged, regardless of
        # whether the event's own KCs (active_event) overlap with them.
        weights *= (1.0 - capture_rate * tag)[:, None]

    return float(mbon_drive(weights, target_code).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=Path(__file__).parent / "data", type=Path)
    parser.add_argument("--weak-eta", type=float, default=0.05)
    parser.add_argument("--strong-eta", type=float, default=0.3)
    parser.add_argument("--capture-rate", type=float, default=1.0)
    parser.add_argument("--tag-decay", type=float, default=0.7)
    parser.add_argument("--gaps", default="0,1,2,4,8")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    neurons_df = pd.read_parquet(args.data_dir / "neurons.parquet")
    matrix = sp.load_npz(args.data_dir / "adjacency.npz")
    projection = np.load(args.data_dir / "pn_projection.npy")
    pn_positions, _ = load_pn_positions(args.data_dir / "neurons.parquet")
    n_total = len(neurons_df)
    kc_mask = (neurons_df["type"].map(cell_class) == "KC").to_numpy()
    kc_positions = np.flatnonzero(kc_mask)
    mbon_positions = np.flatnonzero((neurons_df["type"].map(cell_class) == "MBON").to_numpy())
    weights_template = kc_mbon_weights(matrix, kc_positions, mbon_positions)

    codes = [kc_code_for(s, projection, matrix, pn_positions, n_total, kc_mask, 0.10) for s in DOCUMENT_POOL]

    gaps = [int(g) for g in args.gaps.split(",")]
    rng = np.random.default_rng(args.seed)

    print(f"weak_eta={args.weak_eta}, strong_eta={args.strong_eta}, "
          f"capture_rate={args.capture_rate}, tag_decay={args.tag_decay}, episodes={args.episodes}\n")
    print(f"{'gap':>4} {'rescued':>12} {'control':>12} {'ceiling':>12} {'recovered_fraction':>18}")
    for gap in gaps:
        width = gap + 2
        rescued, control, ceiling = [], [], []
        for _ in range(args.episodes):
            chosen = rng.choice(len(codes), size=width, replace=False)
            target = codes[chosen[0]]
            fillers = [codes[i] for i in chosen[1:1 + gap]]
            event = codes[chosen[-1]]
            shared = dict(
                target_code=target, filler_codes=fillers, event_code=event,
                weak_eta=args.weak_eta, strong_eta=args.strong_eta, tag_decay=args.tag_decay,
            )
            rescued.append(episode_recall(weights_template, capture_rate=args.capture_rate, mode="rescued", **shared))
            control.append(episode_recall(weights_template, capture_rate=0.0, mode="control", **shared))
            ceiling.append(episode_recall(weights_template, capture_rate=0.0, mode="ceiling", **shared))

        r, c, ce = np.mean(rescued), np.mean(control), np.mean(ceiling)
        denom = c - ce
        recovered = (c - r) / denom if abs(denom) > 1e-6 else 0.0
        print(f"{gap:>4} {r:>12.1f} {c:>12.1f} {ce:>12.1f} {recovered:>18.3f}")


if __name__ == "__main__":
    main()
