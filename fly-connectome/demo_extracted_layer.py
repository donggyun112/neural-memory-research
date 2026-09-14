"""Prove the extraction actually works: load the real connectome and
reproduce phase 18's dual-store finding using ONLY fly_memory_layer.py --
no kc_response.py, generalization_probe.py, or capacity_probe.py imported.

Phase 19 demo (see README.md). Reuses data/kc_codes.npz, already computed
by an earlier kc_response.py run, purely as pre-made input vectors -- this
script itself never touches the rest of the pipeline.
"""
from pathlib import Path

import numpy as np

from fly_memory_layer import from_mushroom_body_data_dir

data_dir = Path(__file__).parent / "data"
kc_data = np.load(data_dir / "kc_codes.npz")
sentences = [str(s) for s in kc_data["sentences"]]
codes = kc_data["kc_codes"]

layer = from_mushroom_body_data_dir(data_dir)
state = layer.initial_state(batch_size=1)
print(f"extracted layer: {layer.n_kc} KC x {layer.n_mbon} MBON\n")

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
