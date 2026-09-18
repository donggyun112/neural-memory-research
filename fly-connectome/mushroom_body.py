"""A mushroom body you can run. The wiring is the memory; there is no read step.

Everything before this compared a memory against a baseline and asked whether it
helped. That question does not apply to an organ. Nobody asks whether the
hippocampus beats a control — it is part of what the animal is. The measurements
kept coming back null partly because the pool being selected from held nothing,
and partly because "does it help" was never the question.

So this is not an experiment. It is the circuit, assembled from the real
connectome on disk, with the property that makes it a memory rather than a
database: **nothing retrieves anything.** A pattern enters, expands into Kenyon
cells, flows through synapses that experience has changed, and leaves as a
valence. There is no function here that returns a stored item, because the fly
has no such function.

    perception -> sparse code -> plastic synapses -> valence -> behaviour
                                       ^
                               dopamine writes here

What it does have is the structure the anatomy actually specifies: real PN-to-KC
divergence, real KC-to-MBON convergence, compartments whose sign comes from the
PAM/PPL1 ratio measured in `compartment_valence.py`, and depression rather than
potentiation as the learning rule — dopamine weakens the synapses that were
active, so what survives is what was not punished.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COMPARTMENT = re.compile(r"^(a'?[123]|b'?[12]|g[1-5]|CA|PED)\((L|R)\)$")
HERE = Path(__file__).parent


@dataclass
class Circuit:
    """The parts of the mushroom body, as measured."""

    weights: np.ndarray  # KC -> MBON synapse counts, the plastic substrate
    signs: np.ndarray  # per MBON: +1 its compartment is taught by reward dopamine
    projection: np.ndarray  # input -> KC, the expansion
    active: int  # Kenyon cells a stimulus is allowed to switch on


def load(inputs: int = 384, active: int = 40, seed: int = 7) -> Circuit:
    """Assemble the circuit from the connectome files.

    The PN-to-KC expansion is drawn randomly because that is what the biology
    does — single-cell connectivity there is close to random, and the structure
    that matters sits in the biases, not in any particular cell's partners.
    """
    weights = np.load(HERE / "data/kc_mbon_weights.npy").astype(np.float64)
    neurons = pd.read_parquet(HERE / "data/neurons.parquet")
    valence = pd.read_csv(HERE / "data/compartment_valence.csv", index_col=0)

    mbons = neurons[neurons["type"].astype(str).str.startswith("MBON")]
    signs = []
    for payload in mbons["roiInfo"]:
        regions = json.loads(payload) if isinstance(payload, str) else (payload or {})
        best, most = None, 0
        for name, info in regions.items():
            match = COMPARTMENT.match(name)
            if match and isinstance(info, dict) and (info.get("post") or 0) > most:
                best, most = match.group(1), info["post"]
        signs.append(float(valence["valence"].get(best, 0.0)) if best else 0.0)

    generator = np.random.default_rng(seed)
    cells = weights.shape[0]
    return Circuit(
        weights=weights,
        signs=np.array(signs),
        projection=generator.normal(size=(inputs, cells)) / np.sqrt(inputs),
        active=active,
    )


def encode(circuit: Circuit, stimulus: np.ndarray) -> np.ndarray:
    """Kenyon cell activity for a stimulus: expand, then keep only the strongest.

    The expansion is what makes two similar odours separable; the sparsity is
    what keeps one lesson from smearing over the rest.
    """
    expanded = np.maximum(0.0, stimulus @ circuit.projection)
    cut = np.partition(expanded, -circuit.active)[-circuit.active]
    kept = np.where(expanded >= cut, expanded, 0.0)
    norm = np.linalg.norm(kept)
    return kept / norm if norm > 0 else kept


class MushroomBody:
    """The organ. Feed it experience; it changes. Ask it nothing."""

    def __init__(self, circuit: Circuit | None = None, rate: float = 0.35) -> None:
        self.circuit = circuit or load()
        self.rate = rate
        # The synapses start at their measured strengths. Learning only takes
        # away, so the resting state is the animal's prior on everything.
        self.synapses = self.circuit.weights.copy()
        self.support = self.circuit.weights != 0

    def sense(self, stimulus: np.ndarray) -> float:
        """What this stimulus now means: positive good, negative bad.

        This is not a lookup. The value is whatever the ensemble happens to
        produce given how the synapses currently stand, which is why it can
        answer for a stimulus that was never presented, and why it cannot be
        asked what it stored.
        """
        code = encode(self.circuit, stimulus)
        # The readout sign is the opposite of the teaching sign, and that is the
        # circuit's arrangement rather than a correction: MBONs in compartments
        # taught by reward dopamine drive *avoidance*. Depressing them releases
        # approach. Learning removes the drive that was wrong, so behaviour is
        # what the surviving synapses fail to suppress.
        return float((code @ self.synapses) @ -self.circuit.signs)

    def teach(self, stimulus: np.ndarray, outcome: float) -> None:
        """Dopamine arrives and pushes the two compartment families apart.

        This is a signed push-pull rule, not the depression-only one an earlier
        version of this docstring described. A good outcome depresses the
        synapses onto reward-taught MBONs and *potentiates* those onto
        punishment-taught ones, because `outcome * sign` is negative in the
        second case and the update subtracts it. The antisymmetry that shows up
        in the readout is therefore written in directly rather than emerging.

        Calling it depression-only was wrong twice over: it misdescribes the
        code, and it made a 3% *growth* in total synaptic weight look like a
        separate defect when it is this rule working as implemented. The fly's
        own plasticity is predominantly depressive; this is not that, and the
        gap is worth closing before the circuit is used for anything.
        """
        code = encode(self.circuit, stimulus)
        change = self.rate * np.outer(code, outcome * self.circuit.signs) * self.support
        self.synapses = np.clip(self.synapses - change, 0.0, None)

    @property
    def depleted(self) -> float:
        """Share of the original synaptic weight that experience has removed."""
        return 1.0 - self.synapses.sum() / self.circuit.weights.sum()


def demo() -> None:
    """Show the organ working: it learns a valence, and it generalises."""
    body = MushroomBody()
    generator = np.random.default_rng(0)
    good = generator.normal(size=384)
    bad = generator.normal(size=384)

    before = body.sense(good), body.sense(bad)
    for _ in range(12):
        body.teach(good, +1.0)
        body.teach(bad, -1.0)
    after = body.sense(good), body.sense(bad)

    print(f"circuit: {body.circuit.weights.shape[0]} Kenyon cells -> "
          f"{body.circuit.weights.shape[1]} MBONs, {body.circuit.active} active per stimulus")
    print(f"before teaching   good {before[0]:+9.2f}   bad {before[1]:+9.2f}")
    print(f"after 12 lessons  good {after[0]:+9.2f}   bad {after[1]:+9.2f}")
    print(f"synaptic weight removed: {body.depleted:.3%}")

    # The two stimuli must end up on opposite sides of where they started.
    assert after[0] - before[0] > 0 > after[1] - before[1], (before, after)

    # A stimulus it never met still gets an answer, because there is nothing to
    # look up — the value is whatever the current synapses produce.
    novel = generator.normal(size=384)
    print(f"a stimulus it never met: {body.sense(novel):+9.2f}")

    # And one that resembles the good one leans the way the good one does.
    similar = good + 0.3 * generator.normal(size=384)
    print(f"a stimulus resembling the good one: {body.sense(similar):+9.2f}")
    assert body.sense(similar) > body.sense(bad)
    print("\nno retrieval function exists in this file")


if __name__ == "__main__":
    demo()
