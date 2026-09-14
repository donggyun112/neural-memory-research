"""Feed a handful of real (downloaded, not generated) images through the
real R1-R6/R7/R8 photoreceptor population and one hop into the lamina.

Phase 12/13 of this experiment line (see README.md). R1-R6 uses real
retinotopy from fetch_retinotopy.py (bilinear-sampled at each neuron's real
2D position, PCA-projected from its actual synapse centroid in LA(R)) when
`retinotopy.npz` exists, falling back to phase 12's arbitrary-flatten
version otherwise. R7/R8 stays the phase-12 crude RGB proxy -- those
photoreceptors pass through the lamina without synapsing there (they
terminate in the medulla instead), so LA(R)'s synapses don't cover them.
Real images, not generated ones, so nothing here borrows meaning from a
text-to-image model -- only from whoever photographed and captioned the
source image.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from PIL import Image, ImageFilter
from scipy.ndimage import map_coordinates

from fetch_optic_lobe import photoreceptor_class
from kc_response import propagate, scatter

IMAGES = {
    "coffee": "https://upload.wikimedia.org/wikipedia/commons/4/45/A_small_cup_of_coffee.JPG",
    "dog": "https://upload.wikimedia.org/wikipedia/commons/d/d1/Shaggy_Dog_running.jpg",
    "fire_alarm": "https://upload.wikimedia.org/wikipedia/commons/8/8e/Kobishi_Electric_MSB-63A_fire_alarm_bell.JPG",
}


_CACHE_DIR = Path(__file__).parent / "images_cache"


def fetch_image(url: str) -> Image.Image:
    """Cached to disk by URL -- avoids re-hitting (and getting rate-limited
    by) Wikimedia on every run."""
    cache_path = _CACHE_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".jpg")
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    # Wikimedia rejects the default urllib user agent.
    request = urllib.request.Request(url, headers={"User-Agent": "fly-connectome-experiment/0.1"})
    with urllib.request.urlopen(request) as response:
        data = response.read()

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return Image.open(io.BytesIO(data)).convert("RGB")


def contrast_map(image: Image.Image, blur_radius: float = 15.0) -> np.ndarray:
    """Local brightness minus a heavily blurred version of itself -- a
    high-pass filter approximating what real lamina neurons (L1-L5) compute
    downstream of R1-R6: they respond to local contrast/edges, not raw
    luminance, which is exactly what removes background lighting/composition
    differences that swamped the raw-brightness signal (phase 15)."""
    gray = image.convert("L")
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    g = np.asarray(gray, dtype=float) / 255.0
    b = np.asarray(blurred, dtype=float) / 255.0
    return np.abs(g - b)


def r1_r6_activation(image: Image.Image, n_neurons: int) -> np.ndarray:
    """Contrast map, resized to a small grid, flattened, and padded/truncated
    to n_neurons -- NOT real retinotopy, just enough pixels to differ per
    image."""
    side = int(np.ceil(np.sqrt(n_neurons)))
    contrast = contrast_map(image)
    resized = np.asarray(Image.fromarray((contrast * 255).astype(np.uint8)).resize((side, side)), dtype=float) / 255.0
    flat = resized.flatten()
    return flat[:n_neurons] if len(flat) >= n_neurons else np.pad(flat, (0, n_neurons - len(flat)))


def r1_r6_retinotopic_activation(image: Image.Image, body_ids: np.ndarray, retino_body_ids: np.ndarray, retino_uv: np.ndarray) -> np.ndarray:
    """Bilinear-sample local contrast (not raw brightness) at each neuron's
    real 2D retinotopic position (u, v in [0, 1]) instead of an arbitrary
    flatten order -- this is what lets image structure (edges, shapes)
    survive."""
    body_to_uv = dict(zip(retino_body_ids.tolist(), retino_uv))
    uv = np.array([body_to_uv.get(b, (0.5, 0.5)) for b in body_ids])
    contrast = contrast_map(image)
    h, w = contrast.shape
    rows = uv[:, 1] * (h - 1)
    cols = uv[:, 0] * (w - 1)
    return map_coordinates(contrast, [rows, cols], order=1, mode="nearest")


def color_proxy_activation(image: Image.Image, group: str, n_neurons: int) -> np.ndarray:
    """Crude RGB-channel proxy per R7/R8 subgroup -- not real spectral
    tuning. "y" (yellow-ish, long wavelength) -> red channel mean, "p"
    (pale, short wavelength/UV-adjacent) -> blue channel mean, anything
    else -> green channel mean."""
    arr = np.asarray(image, dtype=float) / 255.0
    channel = 0 if group.endswith("y") else 2 if group.endswith("p") else 1
    return np.full(n_neurons, arr[:, :, channel].mean())


def build_stimulus(image: Image.Image, neurons_df: pd.DataFrame, retinotopy: dict | None = None) -> np.ndarray:
    classes = neurons_df["type"].map(photoreceptor_class)
    n_total = len(neurons_df)
    stimulus = np.zeros(n_total)

    r1r6_mask = (classes == "R1-R6").to_numpy()
    if retinotopy is not None:
        body_ids = neurons_df.loc[r1r6_mask, "bodyId"].to_numpy()
        stimulus[r1r6_mask] = r1_r6_retinotopic_activation(image, body_ids, retinotopy["bodyId"], retinotopy["uv"])
    else:
        stimulus[r1r6_mask] = r1_r6_activation(image, int(r1r6_mask.sum()))

    for group in classes.unique():
        if group in ("other", "R1-R6"):
            continue
        mask = (classes == group).to_numpy()
        stimulus[mask] = color_proxy_activation(image, group, int(mask.sum()))

    return stimulus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=Path(__file__).parent / "data_optic", type=Path)
    args = parser.parse_args()

    neurons_df = pd.read_parquet(args.data_dir / "neurons.parquet")
    matrix = sp.load_npz(args.data_dir / "adjacency.npz")
    other_mask = (neurons_df["type"].map(photoreceptor_class) == "other").to_numpy()

    retinotopy_path = args.data_dir / "retinotopy.npz"
    retinotopy = dict(np.load(retinotopy_path)) if retinotopy_path.exists() else None
    print(f"retinotopy: {'real (fetch_retinotopy.py)' if retinotopy else 'none -- arbitrary flatten order'}\n")

    responses = {}
    for name, url in IMAGES.items():
        image = fetch_image(url)
        stimulus = build_stimulus(image, neurons_df, retinotopy)
        drive = np.maximum(0.0, propagate(stimulus, matrix))
        responses[name] = drive[other_mask]
        print(f"{name}: downstream lamina drive sum={drive[other_mask].sum():.2f}, "
              f"active={int((drive[other_mask] > 0).sum())}/{int(other_mask.sum())}")

    print("\npairwise cosine similarity of downstream lamina response:")
    names = list(responses)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va, vb = responses[a], responses[b]
            cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
            print(f"  {a} vs {b}: {cos:.3f}")


if __name__ == "__main__":
    main()
