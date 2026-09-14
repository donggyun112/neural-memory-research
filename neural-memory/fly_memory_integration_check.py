"""Cross-project check: import fly_memory_layer.py from the sibling
experiments/fly-connectome project and use it here, in neural-memory's own
uv environment/dependencies, not fly-connectome's.

First attempt called from_mushroom_body_data_dir() here and failed --
that loader pulls in pandas, scipy, and (via fetch_mushroom_body.cell_class's
module-level `from neuprint import Client`) neuprint-python too, none of
which this project has or needs. export_memory_weights.py (run once, in
fly-connectome's own env) does that work up front and saves a plain
kc_mbon_weights.npy -- the actual portable artifact. FlyMemoryLayer itself
needs nothing but numpy, and this proves it: no pandas/scipy/neuprint import
anywhere in this file.

Not a permanent integration (fly-connectome isn't packaged for a proper uv
path dependency -- it's a flat script directory), just proof that the
extracted layer (experiments/fly-connectome/fly_memory_layer.py, phase 19)
actually works when attached from outside its own project.
"""
import sys
from pathlib import Path

FLY_CONNECTOME_DIR = Path(__file__).resolve().parents[1] / "fly-connectome"
sys.path.insert(0, str(FLY_CONNECTOME_DIR))

from fly_memory_layer import FlyMemoryLayer  # noqa: E402

import numpy as np  # noqa: E402

data_dir = FLY_CONNECTOME_DIR / "data"
kc_data = np.load(data_dir / "kc_codes.npz")
sentences = [str(s) for s in kc_data["sentences"]]
codes = kc_data["kc_codes"]

weights = np.load(data_dir / "kc_mbon_weights.npy")
layer = FlyMemoryLayer(weights)
state = layer.initial_state(batch_size=1)
print(f"imported from: {FLY_CONNECTOME_DIR / 'fly_memory_layer.py'}")
print(f"running in:    {sys.prefix}  (neural-memory's own venv)")
print(f"layer: {layer.n_kc} KC x {layer.n_mbon} MBON\n")

seen = {}
for sentence, code in zip(sentences, codes):
    drive = float(layer.read(state, code).sum())
    state = layer.write(state, code)
    if sentence in seen:
        pct = 100 * (drive - seen[sentence]) / seen[sentence]
        print(f"  drive={drive:10.2f}  (repeat, {pct:+.1f}%)  {sentence}")
    else:
        print(f"  drive={drive:10.2f}  (first time)         {sentence}")
    seen[sentence] = drive
