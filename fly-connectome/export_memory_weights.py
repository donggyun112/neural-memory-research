"""Export the real KC->MBON weight matrix as a plain .npy file -- the
actual portable artifact, needing no pandas/scipy/neuprint-python to load
elsewhere.

Phase 19 follow-up: attaching fly_memory_layer.py to experiments/neural-memory
found that from_mushroom_body_data_dir() pulls in pandas, scipy, and (via
fetch_mushroom_body.cell_class's module-level `from neuprint import
Client`) neuprint-python too -- none of which that project has, and none of
which FlyMemoryLayer itself needs. This script does the pandas/scipy/neuprint
work once, here, and saves just the numbers.
"""
from pathlib import Path

import numpy as np

from fly_memory_layer import from_mushroom_body_data_dir


def main() -> None:
    data_dir = Path(__file__).parent / "data"
    layer = from_mushroom_body_data_dir(data_dir)
    out_path = data_dir / "kc_mbon_weights.npy"
    np.save(out_path, layer.weights)
    print(f"{layer.n_kc} KC x {layer.n_mbon} MBON -> {out_path}")


if __name__ == "__main__":
    main()
