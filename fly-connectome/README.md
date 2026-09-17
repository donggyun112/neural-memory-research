# fly-connectome

Independent experiment line, separate from `experiments/neural-memory`. Goal: a memory
network built directly on the *real* Drosophila mushroom body circuit -- actual connectome
wiring, not a mechanism borrowed from it -- that takes text as input. The neural-memory line
deliberately distills connectome *mechanisms* (sparse expansion ratio, opponent dopaminergic
channels, per-compartment decay) into a graph-free fast-weight model (see its `RESEARCH.md`,
"No graph or external store"). This line does the opposite: keep the actual graph, and let
its own real KC->MBON synapses be the thing that changes with experience.

Going one step at a time: text -> real PN neurons -> real KC layer -> a first (weak, no
ground truth yet) stimulus-scoring attempt -> real write/recall plasticity on KC->MBON,
which is where this stopped treating "compare sentences" as the goal and started building
the actual memory mechanism.

## Status overview (35 phases so far)

- **Text/memory line, the main thread (phases 1-8, 17-28, 34, 36):** real mushroom body circuit,
  real write/recall memory, generalization, capacity, and an opponent-channel valence mechanism
  that works correctly given an arbitrary sign split (phase 34). **Phase 36 overturns phase 25's
  choice of mechanism:** its comparison metric contained each item's own write, so it rewarded
  writing shallowly, and it never read an item the circuit had not seen. Scored on separating
  written from unwritten documents, the best mechanism is a single saturating store (`eta` near
  1) and phase 25's habituated dual-store is the worst above 16 items. Phase 36 also gives this
  line its first capacity number: 0.98 AUC at 32 documents, 0.63 at 256. **Phase 37 then predicts
  that number from two integers:** the memory is a Bloom filter over Kenyon cell identities, and a
  parameter-free formula in `k` (cells active) and `m` (cells total) matches measurement within
  noise once code correlation is removed. Correlation is the entire deviation, and it both lowers
  peak capacity and softens the ceiling. Measured sparsity optimum 0.02-0.05, against a real
  APL-enforced 5-10 percent.
- **Extracted as a reusable layer (phases 19, 26):** `fly_memory_layer.py`, plain numpy,
  proven to run standalone and attached inside `experiments/neural-memory`'s own venv.
- **Vision line (phases 11-16, 27):** real photoreceptors and retinotopy; interference
  tracks *measured* similarity correctly once human topic labels stop being trusted as a
  stand-in for it (phase 27).
- **Sound line (phases 9, 29-32):** real Johnston's organ input, real write/recall memory,
  and real content discrimination (interference tracks cosine similarity), after finding and
  fixing two real bugs a small/convergent circuit exposed that the larger text/vision
  populations had hidden (single-scalar input, phase 30; magnitude-blind write, phase 32).
- **Odor grounding (phases 10, 33):** 7 real, literature-specific odorant-to-glomerulus
  pairings, deliberately not generalized further without stronger evidence per pairing.
- **Real external data (phases 21, 35):** the pipeline runs cleanly on real LongMemEval
  document embeddings from a different text encoder; confirmed twice now (episode grouping,
  then real `question_type` labels) that no available label substitutes for *measuring*
  similarity directly -- only phases 6/27's direct measurement showed real discrimination.
- **What's NOT real, stated plainly:** the fly does not understand text, images, or arbitrary
  concepts -- see the section immediately below. Only the wiring and its measurable
  computational properties are real.

## What's actually real here, and what isn't

The fly does not know what "coffee" is. Three pieces, and only one of them is fly biology:

1. **Semantic similarity** ("a loud alarm" ~ "the fire alarm") comes entirely from
   `all-MiniLM-L6-v2`, a model trained on human text. Nothing to do with the fly.
2. **Which of the 138 real PN neurons a meaning lands on** is a *random* matrix
   (`pn_projection.npy`) we made up. There is no real odorant-receptor-to-glomerulus mapping
   here -- "this PN represents coffee" is not a true statement about anything.
3. **The only real fly biology in this pipeline** is the synaptic wiring itself: the actual
   PN->KC->MBON connection structure and weights pulled from MaleCNS v1.0.

So what phases 5-7 actually tested is: does routing an arbitrary (if semantically
structured) input through *this real wiring topology* produce useful properties -- KC-overlap
generalization, topic-sensitive interference, capacity limits -- and not "does the fly
understand text." Phase 8 (below) is that control.

## Data source

**MaleCNS v1.0** (Janelia/HHMI + Google Research + Cambridge + MRC LMB; Cell, Sept 2026):
166,700 neurons, 125M+ synapses, the full CNS (brain + ventral nerve cord) of the male fly,
released June 2026 and queryable through the same public `neuprint.janelia.org` server as
the older hemibrain dataset. This is the most complete public connectome available, and
supersedes hemibrain (2020, central-brain only, ~25k neurons), which is still selectable via
`--dataset hemibrain:v1.2.1` if needed. Cell-class name patterns (`KC`, `MBON`, `PAM`/`PPL`,
`*PN`) come from hemibrain-era naming; verify they still match MaleCNS's own type/ROI names
against the first fetch's printed summary before trusting the counts.

## Phase 1 (this step): load and verify the real circuit

`fetch_mushroom_body.py` pulls every traced neuron touching the right mushroom body
(`MB(R)`) from neuPrint, plus the synaptic connections between them, and saves a sparse
weighted adjacency matrix. No dynamics, no text input yet -- just getting real data in and
checking the counts look like the published circuit (about 50 PN, about 2,000 KC per
hemisphere, per `experiments/neural-memory/RESEARCH.md`).

### Setup

1. Register for a free neuPrint account and token at https://neuprint.janelia.org
2. `export NEUPRINT_TOKEN=...`
3. `uv sync`

### Run

```
uv run python fetch_mushroom_body.py
uv run pytest
```

Output goes to `data/adjacency.npz` (scipy sparse matrix) and `data/neurons.parquet`
(neuron metadata, same row order as the matrix).

**Result (MaleCNS v1.0, MB(R)):** KC 2,053 / DAN 333 / PN 138 / MBON 75, 1,016,354 synapse
edges. KC count lines up with the published ~2,000 per hemisphere.

## Phase 2 (this step): map text onto the real PN neurons

`map_text_to_pn.py` splits input text into sentences, embeds each with a small
sentence-transformer (`all-MiniLM-L6-v2`), and projects each embedding onto the circuit's
138 real PN neurons through a fixed random matrix (persisted at `data/pn_projection.npy` so
the same sentence always lands on the same PN pattern). Firing rates are clipped at zero.
Still no KC dynamics and no stimulus-point definition -- just a PN activation vector per
sentence.

### Run

```
uv run python map_text_to_pn.py --text "Sentence one. Sentence two."
# or: --text-file some.txt
```

Output goes to `data/pn_activations.npz` (`activations`, `pn_positions`, `pn_ids`, `sentences`).
`pn_positions` are row/column indices into `adjacency.npz` (matrix order, not sorted by
bodyId) -- that's what phase 3 uses to inject activation into the real graph.

## Phase 3 (this step): propagate PN activity to the KC layer

`kc_response.py` scatters each sentence's PN activation into a full-length vector over every
neuron in the circuit, propagates it one synaptic hop through the real weighted adjacency
matrix, then keeps only the top ~10% of KC drive (`keep_ratio`, matching Lin et al. 2014's
5-10% sparse coding), zeroing the rest as a stand-in for APL inhibition.

### Run

```
uv run python kc_response.py
```

Output goes to `data/kc_codes.npz` (`kc_codes`, `sentences`).

**Result:** 3 test sentences -> 2,053 KC, 205 active each (fixed by `keep_ratio`), with
distinct mean drive per sentence (0.457 / 0.572 / 0.594) -- the circuit is producing
different KC codes for different sentences, not a degenerate constant response.

## Phase 4: a first stimulus-scoring attempt (weak, no ground truth)

`stimulus_points.py` scores each sentence by 1 - mean Jaccard overlap of its active-KC set
against every other sentence in the same input (sparse-code overlap, not raw-magnitude
cosine, following how the olfactory literature compares KC codes).

**Result on the 3-sentence test text:** coffee (0.864) > fire alarm (0.841) > typing (0.795).
This does not match any intuition about which sentence is "more salient" -- and there's no
ground truth here to say it should. With only 3 sentences comparing only to each other, this
number is close to noise; it needs a baseline corpus of "ordinary" sentences to mean
anything, which isn't built. Left as a working but unvalidated tool
(`uv run python stimulus_points.py`, reads `data/kc_codes.npz`); not the main line anymore --
see phase 5.

## Phase 5 (this step): real write/recall memory on KC->MBON

`memory.py` is the actual memory mechanism, not a score. It slices the real KC->MBON
synaptic weights out of the loaded connectome (2,053 KC x 75 MBON here), then:

- **write**: depress the KC->MBON weight rows for the currently-active KC set, by a fixed
  fraction `eta` (default 0.3) -- mirrors Owald et al. 2015: reinforcement depresses mushroom
  body output synapses. No per-compartment valence sign is invented (MBON type names alone
  don't give us one); every write depresses, unconditionally.
- **recall**: MBON drive = active-KC-weighted sum over the current (possibly-depressed)
  weights.

Falsifiable claim: recalling the exact same KC code twice, with its own first write in
between, must show a lower second-time drive -- the weights it draws on were strictly
reduced. Confirmed on real data: reusing "The fire alarm rang loudly in the empty hallway."
gave drive 109,716 the first time and 64,212 the second (-41%), while three different
sentences never seen before stayed in the 109k-136k range.

### Run

```
uv run python memory.py
```

## Phase 6: does it generalize, or only recognize exact repeats?

`generalization_probe.py` writes one sentence, then checks recall on a *paraphrase*
(different words, same event) against an unrelated control, both never written before.

**Result:** paraphrase ("A loud alarm blared through the deserted corridor.") shared 118/205
active KCs with the original ("The fire alarm rang loudly in the empty hallway."); the
unrelated control shared only 34/205. After writing the original, the paraphrase's recall
dropped 18.1%, the control's only 5.2%. The memory isn't just recognizing exact strings --
real KC overlap, produced by a real semantic text encoder plus real connectome wiring, makes
similar-but-not-identical events partially interfere, in proportion to how similar they are.

### Run

```
uv run python generalization_probe.py
```

## Phase 7: capacity -- does interference stay contained, or does it accumulate?

`capacity_probe.py` writes 12 sentences across 4 topics (3 near-paraphrases each) in
sequence, snapshotting every sentence's recall after every single write.

**Result:** one-shot interference from writing one sentence onto every *other* sentence
averaged -10.31% for same-topic pairs (n=24) vs -6.55% for different-topic pairs (n=108) --
same-topic is worse, but different-topic interference is far from zero. More importantly,
the very first sentence written keeps losing recall as purely unrelated sentences get
written after it, with no floor: -30.0% after its own write, -50.1% after 3 more, -61.2%
after 6 more, -71.2% after all 11: **-30.0% -> -50.1% -> -61.2% -> -71.2%**.

This is a real limitation, not a bug: unconditional depression on every active KC has no
capacity control, so writing anything erodes everything a little, and it compounds without
bound. This is exactly the missing piece `RESEARCH.md` already names --
**competitive allocation under a roughly fixed budget** (Jeong et al. 2021): potentiated
inputs get preferentially recruited while total recall-active population stays about
constant, so strengthening one trace should cost others in a bounded way, not for free and
not without limit.

### Run

```
uv run python capacity_probe.py
```

## Phase 7b: does a fixed plasticity budget fix it?

`memory.write_budgeted()` does the same depression, then rescales the *entire* weight
matrix to conserve total synaptic weight -- the simplest reading of "roughly constant
recall-active population" (Jeong et al.). Run with `capacity_probe.py --write-mode budgeted`.

**Result:** better, not solved. Load curve for the first item: **-27.7% -> -43.7% -> -52.8%
-> -60.6%** after 0/3/6/11 writes, versus -30.0% / -50.1% / -61.2% / -71.2% unbudgeted --
roughly 10 points less erosion by the end, but still a steady, unbounded decline, not a
floor. Same-topic vs different-topic interference also separated a bit more cleanly
(-7.95% vs -4.09%, a 1.94x ratio, versus 1.57x unbudgeted). A single global rescale is too
blunt an instrument to actually cap interference -- it partially compensates by nudging
every untouched synapse up a little, which helps on average but doesn't stop targeted
drift on any specific pair.

## Phase 8: real connectome vs. a matched random control

`kc_response.random_rewire()` builds a null model: same edge count and same weight values as
the real matrix, but every edge's target is a uniformly random neuron. Run the same phase-7
battery on it with `capacity_probe.py --random-control`.

**Result, real vs. random (same-topic / different-topic interference, and load curve after
11 writes):**

| | same-topic | diff-topic | ratio | load @ 11 writes |
| --- | --- | --- | --- | --- |
| real | -10.31% | -6.55% | 1.57x | -71.2% |
| random | -12.80% | -8.62% | 1.49x | -81.2% |

Honest reading: **topic discrimination is barely different from random** (1.57x vs 1.49x) --
most of the "similar sentences interfere more" behavior comes from the semantic embedding
plus sparse-coding math in general, not from the real connectome's specific wiring. What the
real wiring *does* give, versus a matched random graph, is somewhat less catastrophic
degradation under load (-71% vs -81% after 11 unrelated writes). So the fly does not
"understand" text (see the section above), and swapping in the real topology instead of a
random one of the same size doesn't add much semantic discrimination either -- its
measurable contribution here is milder, more graceful capacity loss.

### Run

```
uv run python capacity_probe.py --random-control
```

## Phase 9: a circuit with a real transduction pathway (sound, not text)

Text has no real path into a fly nervous system -- phase 8 measured the cost of that. Near-
field sound does: physical vibration -> antenna -> Johnston's organ (JO) mechanoreceptor
neurons -> AMMC, a real sensory pathway, so a "fly embedder" for sound wouldn't need an
invented random projection the way `map_text_to_pn.py` needed one for text.

`fetch_johnstons_organ.py` pulls AMMC(R) from MaleCNS v1.0 (mirrors `fetch_mushroom_body.py`;
both now share `connectome_io.py` for the fetch/adjacency/save logic). Real named JO
subgroups exist in the `type` field. Per Kamikouchi et al. 2009, JO-A/JO-B are
sound/vibration-responsive and JO-C/JO-E are gravity/wind-responsive -- at the group level
only; finer subtype tuning (JO-B1_a vs JO-B1_b, etc.) is not something this project can
verify.

**Result (AMMC(R) only):** JO-E 113, JO-F 32, JO-C 20, JO-B 11, JO-A 2, everything else
(local/downstream AMMC neurons, not JO afferents themselves) 2,631; 326,667 synapse edges.
The sound-responsive groups (A+B) are only **13 neurons total** here -- far fewer than the
138 PN used for text. Possibly AMMC(L) has more, or these subtypes aren't fully proofread yet
in this new dataset; not resolved.

### Run

```
NEUPRINT_TOKEN=... uv run python fetch_johnstons_organ.py
```

Output goes to `data_ammc/adjacency.npz` and `data_ammc/neurons.parquet` (separate from the
mushroom body's `data/`, since it's a different circuit).

## Phase 10: a grounded text->PN converter (small, only for known odorants)

`odor_embedder.py` replaces the random fallback with a *real* mapping when a sentence names
one of a small, deliberately conservative set of specific, well-studied odorants: PN `type`
names in this connectome are already glomerulus-named (e.g. `DA2_lPN`), and a handful of
odorant-to-glomerulus pairings are well established in the literature (geosmin -> DA2,
Stensmyr et al. 2012; acetic acid/vinegar -> DM1 and ethanol -> DM2, Hallem & Carlson 2006;
CO2 -> V, Suh et al. 2004; the cVA pheromone -> DA1, Kurtovic et al. 2007). Confirmed against
the loaded MB(R) circuit: every one of these glomeruli has a matching real PN type. Anything
that doesn't name one of these odorants falls back to `map_text_to_pn.py`'s random
projection, and the tool reports which method was used per sentence.

**Result:** "The soil smelled of geosmin after the rain." -> `grounded:DA2`, activating the
real 5 DA2_lPN neurons. "She poured vinegar into the salad dressing." -> `grounded:DM1`,
activating the real 1 DM1_lPN neuron. Everything else (fire alarm, coffee) -> unchanged
`random-fallback`.

### Run

```
uv run python odor_embedder.py --text "..."
uv run python kc_response.py
uv run python memory.py
```

`odor_embedder.py` now writes the same `pn_activations.npz` shape `map_text_to_pn.py` does
(plus a `methods` field), so it's a drop-in replacement -- `kc_response.py`/`memory.py`
needed no changes.

**First (uncalibrated) result:** grounded sentences showed *much* higher MBON drive than
random-fallback ones (geosmin 242,623 / vinegar 534,798 vs fire-alarm 109,716 / coffee
142,196) -- not a discovery about smell salience, a scale mismatch: grounded activation put
a fixed magnitude of 1.0 on very few real PNs (5, or even 1), while the random path spreads a
much smaller mean magnitude across 65.

**Fix:** `reference_pn_magnitude()` measures the random path's typical per-neuron activation
on neutral filler text and uses that instead of the fixed 1.0. **Calibrated result:** geosmin
9,368 / vinegar 20,650 vs fire-alarm 109,716 / coffee 142,196 -- grounded sentences are now
*lower*, which is the expected, honest direction: putting the same per-neuron magnitude
through far fewer real channels (1-5 vs 65) naturally carries less total signal downstream.

## Phase 11: a circuit with a real transduction pathway (vision, general-purpose)

Prompted by DOOMFLY (github.com/nftechie/doomfly, Alex Wormuth) -- a public project already
mapping ViZDoom frames onto this exact connectome's photoreceptors (3,335 R1-R6 brightness +
811 R8 color inputs) and real descending neurons back to game controls, with damage driving
real PPL101 dopamine cells as reinforcement onto 4,184 KC->MBON11 synapses. Notably it
reports **no demonstrated learned survival after 6 iterations** -- a sober data point for
what to expect from this style of experiment on this connectome.

Vision is a stronger "real embedder" candidate than smell for arbitrary content: light
intensity -> photoreceptor response is real biophysics, and it doesn't need a curated
per-concept table the way odor grounding does -- any rendered image goes in. The catch:
photoreceptor transduction being real doesn't mean the fly's visual system recognizes
*objects* -- it's tuned for motion, contrast, and color opponency, not "coffee cup"
detection. Rendering "coffee" as an image and feeding it in still borrows meaning from
whatever rendered the image, same as the text embedder does -- it moves the borrowed-meaning
step, it doesn't remove it.

`fetch_optic_lobe.py` pulls `LA(R)` (lamina, where R1-R8 axons terminate) from MaleCNS
v1.0 -- ~7k neurons, comparable in size to the mushroom body circuit; the next stage,
`ME(R)` (medulla), is ~43k, too big for a first step.

**Result:** 893 R1-R6, 468 R7/R8 subtype neurons (color/UV/dorsal-rim), 5,384 downstream
lamina neurons, 47,731 synapse edges.

### Run

```
NEUPRINT_TOKEN=... uv run python fetch_optic_lobe.py
```

Output goes to `data_optic/adjacency.npz` and `data_optic/neurons.parquet`.

## Phase 12: real (downloaded, not generated) images through the real photoreceptors

`image_to_photoreceptors.py` sidesteps the "what generates the image" question from phase 11
entirely -- it downloads a handful of real photos (Wikimedia Commons, CC-licensed: a coffee
cup, a running dog, a fire alarm bell) instead. Grayscale, resized and flattened onto the 893
R1-R6 neurons (brightness), RGB channel means onto the R7/R8 groups (color proxy) -- **no
retinotopy**: photoreceptor order is whatever the neuron table gives, not real visual-field
position, so spatial structure (edges, shapes) is not preserved, only rough brightness/color
statistics. Propagated one hop through the real `LA(R)` adjacency to the downstream lamina
neurons (L1-L5 etc.).

**Result:** the pipeline runs end-to-end on real data and produces genuinely different
downstream responses per image, but discrimination is weak -- pairwise cosine similarity
0.907-0.940 (coffee vs dog 0.907, coffee vs fire alarm 0.907, dog vs fire alarm 0.940). Not
surprising: without retinotopy, three different photos mostly collapse to "similar overall
brightness/color," which is honestly most of what this crude an encoding could show.

### Run

```
uv run python image_to_photoreceptors.py
```

## Phase 13: real retinotopy instead of arbitrary photoreceptor order

`fetch_retinotopy.py` computes each R1-R6 neuron's real 2D position: fetches its actual
synapses within `LA(R)` from neuPrint, takes the centroid, and PCA-projects the 3D
centroids to 2D (the lamina is a roughly planar sheet, so this recovers something close to
the real retinotopic layout without external anatomical annotation). Saved as
`data_optic/retinotopy.npz`.

R7/R8 could not get the same treatment: fetching their synapses in `LA(R)` returned zero
matches. That's real biology, not a bug -- R7/R8 axons pass *through* the lamina without
terminating there; they synapse in the medulla (`ME(R)`, ~43k neurons, not fetched here).
They stay on phase 12's crude uniform color proxy.

**Result:** 893/893 R1-R6 neurons got a real position. Planarity check (3rd PCA singular
value / 2nd): 0.309 -- reasonably flat but not perfectly planar, consistent with the
compound eye's actual curvature. `image_to_photoreceptors.py` now bilinear-samples each
neuron's real position instead of flattening in table order, when `retinotopy.npz` exists.

**Effect on discrimination:** cosine similarity dropped for the visually distinct pair
(fire alarm vs. the other two) while staying flat for the visually similar pair:

| pair | phase 12 (arbitrary order) | phase 13 (real retinotopy) |
| --- | --- | --- |
| coffee vs dog | 0.907 | 0.906 |
| coffee vs fire alarm | 0.907 | 0.775 |
| dog vs fire alarm | 0.940 | 0.802 |

Real position information measurably helped, without any change to the images or the
encoding math -- but this is still one hop through a small, spatially-shallow relay (the
lamina), at 893 "pixels," so read it as "retinotopy helps" rather than "the fly recognizes
these photos."

### Run

```
NEUPRINT_TOKEN=... uv run python fetch_retinotopy.py   # once, produces retinotopy.npz
uv run python image_to_photoreceptors.py               # picks it up automatically
```

## Phase 14: write/recall memory on the vision circuit

First attempt used raw R1-R6 activation directly as the write/recall code, reusing
`memory.write()`/`mbon_drive()` on the real R1-R6 -> downstream-lamina synapses. **This was
wrong and gave a fake result**: real photos leave ~100% of R1-R6 nonzero (unlike text's PN
activation, which is naturally ~50% sparse from ReLU on random projections), so writing *any*
image depressed essentially the whole matrix. Coffee, dog, and fire alarm all dropped by
exactly -30.0%, written or not -- a global effect, not memory.

**Fix**, mirroring the text pipeline's actual shape: R1-R6 (dense, PN-equivalent) propagates
one hop to the downstream lamina ("other" -- L1-L5, amacrine cells etc., KC-equivalent),
*then* gets sparse-coded (top 10%, `kc_response.sparse_code`) -- only the sparse code is used
for write/recall, against the real "other -> other" lamina recurrent synapses (the honest
MBON-equivalent available without fetching the medulla).

**Corrected result:** active-set overlap between different images is high (coffee/dog 0.666,
coffee/fire_alarm 0.578, dog/fire_alarm 0.546 -- this circuit's discrimination is weak, as
phase 12/13 already showed). Writing coffee now gives a **graded**, not uniform, effect:
coffee itself -30.0%, dog (higher real overlap, 0.666) -25.9%, fire_alarm (lower real
overlap, 0.578) -25.1%. The ordering tracks the real overlap -- this is the same
overlap-proportional-interference signature phase 6 found for text, now confirmed on vision
too, once the sparse-coding step was actually there to make it meaningful.

**Follow-up with more images (2 per topic, 3 topics -- coffee, dog, alarm; Wikimedia
Commons, cached locally in `images_cache/` to avoid re-hitting/rate-limiting their
servers):** same-topic vs. different-topic interference is now indistinguishable --
**-25.75% (n=6) vs -25.97% (n=24)**. Pairwise KC-set overlap sits in a narrow 0.55-0.74 band
regardless of topic (e.g. coffee-cup vs coffee-beans, same topic, overlaps 0.673; coffee-cup
vs the unrelated smoke detector overlaps *higher*, at 0.730). More data didn't reveal
discrimination that a smaller sample was hiding -- it confirmed there isn't any at this
encoding's resolution. This is worse than even text's random-graph control (phase 8,
1.49x-1.57x same/diff ratio) -- 893 "pixels," a crude non-retinotopic color proxy, and one
propagation hop aren't enough structure for real image content to separate.

### Run

```
uv run python vision_memory_probe.py
```

## Phase 16: local contrast instead of raw brightness

Real lamina neurons (L1-L5) are contrast/edge channels, not raw-luminance channels -- so
`contrast_map()` replaces raw grayscale sampling with local brightness minus a heavily
blurred version of itself (a high-pass filter) before retinotopic sampling. The intent: stop
letting background lighting/composition dominate the signal.

**Result:** overlap spread widened a lot (0.330-0.708, versus a narrow 0.546-0.740 band
before) -- the encoding is now clearly sensitive to actual image content. But same-topic vs.
different-topic interference is *still* not reliably separated (-20.24% vs -19.00%).

**Why, checked directly:** bell vs. smoke_detector (same topic) overlaps 0.679 -- both are
photos of round objects mounted on a wall, and they really do look alike. Cup vs. beans (same
topic) overlaps only 0.425, and shaggy vs. golden (same topic, two different dog photos) only
0.452 -- a coffee cup and a pile of beans don't share visual structure just because a human
calls both "coffee," and two dog photos in different poses/framing don't automatically share
edge structure either. **The topic labels encode human semantic category, and this circuit
only ever sees contrast/shape -- those are different axes, and phase 16 didn't fail, it
exposed that mismatch.** A fair test would need topics defined by visual similarity (same
pose, framing, and silhouette), not by what a human would call the same thing.

## Phase 17: a properly localized competitive budget -- and the trade-off it exposes

`memory.write_competitive()` implements `RESEARCH.md`'s actual licensed shape (`a =
sparse_competition(...)`, `sum(a) = 1`) instead of phase 7b's blunt global rescale: a fixed
depression budget is shared across the currently-active KC set, each KC getting a share
proportional to its own activation, so a write's total damage no longer grows with how many
KCs happen to be active. All three write modes (`unbudgeted`, `budgeted`, `competitive`) are
now selectable via `--write-mode` on both `memory.py` and `capacity_probe.py`.

**Capacity result (phase 7's battery, load after 11 writes):** unbudgeted -71.2%, budgeted
(global rescale) -60.6%, **competitive -0.7%** -- the erosion problem is essentially solved.

**But recognition memory nearly vanishes with it.** Re-running phase 5's exact-repeat test
under `--write-mode competitive`: the fire-alarm sentence's recall goes from 109,716 to
109,444 on its second exposure -- **-0.25%**, versus unbudgeted's validated **-41%**. With
eta=0.3 spread over ~205 active KCs, each one only takes a ~0.15% hit, too small for a single
repeat to show up. Fixing capacity by spreading a fixed budget over every active KC on every
write also spreads away the very effect phase 5 was built to demonstrate.

This isn't a dead end, it's a real result: a single flat competitive budget can't satisfy
both properties (bounded capacity AND a legible per-item recognition signal) at once, which
is likely exactly why `RESEARCH.md` doesn't stop at one global budget -- it separately
licenses parallel stores with different time constants (measurement 4) and a synaptic-tagging
rescue window (measurement 7). Neither is built yet; phase 17 is the concrete evidence for
why they're needed, not just literature saying so.

### Run

```
uv run python capacity_probe.py --write-mode competitive
uv run python memory.py --write-mode competitive
```

## Phase 18: two stores, different decay, read as a sum

`dual_store_probe.py` implements `RESEARCH.md` measurement 4 directly: a fast store
(eta=0.5, carries the recognition signal) and a slow store (eta=0.03, resists erosion),
both written on every event, recall = their sum.

**Result -- a real middle ground between phase 17's two extremes:**

| | exact-repeat recognition | load after 11 writes |
| --- | --- | --- |
| unbudgeted (phase 5/7) | -41% | -71.2% |
| competitive (phase 17) | -0.25% | -0.7% |
| **dual (fast+slow), phase 18** | **-26.5%** | **-49.6%** |

Same-topic vs. different-topic interference also came back (-5.82% vs -3.56%, ~1.64x ratio,
comparable to unbudgeted's 1.57x). Neither property was sacrificed to get the other -- this
is the first mechanism in this line that gives a legible recognition signal *and*
meaningfully better capacity behavior than the original unbudgeted write, at the same time.
`eta` values (0.5 fast / 0.03 slow) are a first pass, not tuned.

### Run

```
uv run python dual_store_probe.py
```

## Phase 19: extracted as a standalone memory layer

`fly_memory_layer.py` -- just the validated mechanism (phase 18's dual fast/slow store),
with the connectome-fetching and text/image front-end entirely stripped away.
`FlyMemoryLayer` needs only a KC x MBON weight matrix (plain numpy) to construct; nothing
about neuPrint, embeddings, or sentence/image encoding. `from_mushroom_body_data_dir()` is a
convenience loader for this repo's cached fetch (no token/network needed, just the local
`.parquet`/`.npz` files).

**Interface refined to sit next to `experiments/neural-memory/neural_memory/*.py`** (numpy,
not torch -- that line's models are the reason for the shape, not a reason to add a torch
dependency here): a frozen `FlyMemoryState` dataclass (mirrors `MemoryState`/
`FastWeightState`) carries the per-episode weights; `write(state, kc_code) -> FlyMemoryState`
returns a *new* state instead of mutating in place, matching that project's
`write(...) -> State` convention; and both `write`/`read` take a batch-first `kc_code` of
shape `[batch, n_kc]` (a plain 1D code is accepted too, treated as batch size 1), since every
model over there processes a batch dimension.

**Proof it's really independent:** `demo_extracted_layer.py` reproduces phase 18's
recognition result using *only* `fly_memory_layer.py` -- it imports nothing from
`kc_response.py`, `memory.py`, `generalization_probe.py`, or `capacity_probe.py`. Real
connectome weights, loaded from the cached data:

```
drive=219,432  (first time)   The fire alarm rang loudly in the empty hallway.
drive=256,598  (first time)   She kept typing her report, unaware of the noise.
drive=278,486  (first time)   A cup of coffee sat cold on the desk.
drive=145,060  (repeat, -33.9%)   The fire alarm rang loudly in the empty hallway.
```

Answers the question this phase was built to answer: yes, the memory-role network can be
pulled out and reused as a plain layer by anything that can supply a KC-space code vector,
independent of how that vector was produced.

**Actually attached to `experiments/neural-memory` and run there, in its own venv.** First
attempt called `from_mushroom_body_data_dir()` from that project and failed --
`fetch_mushroom_body.cell_class`'s module-level `from neuprint import Client` drags in
pandas, scipy, and neuprint-python, none of which `neural-memory`'s environment has or
`FlyMemoryLayer` itself needs. Fixed with `export_memory_weights.py`: run once here (where
those packages exist) to save the real KC->MBON matrix as a plain `data/kc_mbon_weights.npy`.
`experiments/neural-memory/fly_memory_integration_check.py` then loads that `.npy` directly
(`np.load`, nothing else) and reproduces the exact same result inside `neural-memory`'s own
`.venv` (numpy + torch only):

```
running in: experiments/neural-memory/.venv
layer: 2053 KC x 75 MBON
drive=219,432 (first time) / 256,598 (first time) / 278,486 (first time) / 145,060 (repeat, -33.9%)
```

### Run

```
uv run python demo_extracted_layer.py

# cross-project attach check:
uv run python export_memory_weights.py                                # in fly-connectome, once
cd ../neural-memory && uv run python fly_memory_integration_check.py   # in neural-memory
```

## Phase 26: extracted layer upgraded to phase 25's mechanism

`FlyMemoryLayer.write()` (phase 19) shipped with phase 18's plain dual-store. Updated to
phase 25's habituated dual-store (each store's eta now scaled by that row's own remaining
freshness) -- the best mechanism found in this line, not just the first one extracted.
Re-verified both standalone (`demo_extracted_layer.py`) and cross-project
(`fly_memory_integration_check.py`, run inside `neural-memory`'s own venv) still reproduce a
real recognition drop end to end; existing `FlyMemoryLayer` tests needed no changes beyond
one new habituation check, since freshness is 1.0 on a never-touched row -- the first write
from a fresh state is numerically identical to the old flat-eta rule.

## Phase 20: the synaptic-tagging rescue window (measurement 7)

Ports the design already validated in `experiments/neural-memory/analyze_tagging_window.py`
(git log: "a fading tag produces the rescue window the corpora could not show") onto our real
KC->MBON depression, instead of reinventing the measurement. That project's mechanism is an
associative outer-product store (cosine-to-stored-value as the readout); ours is
depression-only, so "better memory" here means *more* depression, not a higher cosine to a
stored value -- the translation: a weak write banks its unrealized depression
(`strong_eta - weak_eta`) as a per-KC tag; the tag decays every subsequent write; a later
strong event at *any*, possibly unrelated, KCs releases a capture pulse that applies extra
depression to whatever is still tagged -- content-free, exactly as the reference notes ("the
event contributes only magnitude"). `tagging_rescue_probe.py` runs this on the real MB(R)
connectome across a pool of 20 real sentences.

**Result -- a clean, monotonic rescue window:**

| gap (intervening writes) | recovered_fraction |
| --- | --- |
| 0 | 0.665 |
| 1 | 0.465 |
| 2 | 0.326 |
| 4 | 0.160 |
| 8 | 0.038 |

An immediate strong event recovers 66.5% of the way from "written weakly and left alone"
back to "written strong from the start"; by 8 intervening writes that's down to 3.8%. Also
confirmed content-free in a unit test: rescue happens even when the strong event's own KCs
share nothing with the target's.

### Run

```
uv run python tagging_rescue_probe.py
```

## Phase 21: real LongMemEval document embeddings, not our own sentences

`longmemeval_capacity_probe.py` re-runs the phase 7 interference battery, but on real document
embeddings from `experiments/neural-memory/artifacts/longmemeval_deferred.pt` (candidates,
384-dim, a different text encoder than our own `all-MiniLM-L6-v2`) instead of hand-picked
sentences -- through the same real PN->KC pipeline (the existing 384-dim `pn_projection.npy`
happened to fit without modification).

**Deliberately not attempting `neural-memory`'s `gist_recovery`/`surface_recovery`
(`analyze_load_and_gist.py`):** that metric asks whether stored *content* (a value vector) can
be reconstructed, via an associative delta-rule store. Our KC->MBON mechanism is
depression-only -- there is no stored value vector to reconstruct, only interference/
recognition (can it tell something has been seen before). That's a different kind of memory,
not a worse copy of the same one, so gist/surface recovery isn't a well-defined number to
compute for it.

**Result, at 4 and 20 real episodes (same-episode vs. different-episode, "8 candidates per
episode" being LongMemEval's own grouping):**

| episodes | same-episode | different-episode |
| --- | --- | --- |
| 4 | -6.95% (n=224) | -7.27% (n=768) |
| 20 | -2.56% (n=1120) | -2.75% (n=24320) |

No discrimination, at either scale -- but this is likely the wrong comparison, not a new
finding against the mechanism: LongMemEval's 8 candidates per episode are distractor
*sessions* for a QA task, sampled to make the task hard, not a hand-picked cluster of
topically similar text the way phase 7's own sentences were. "Same episode" here isn't a
stand-in for "topically similar" the way our own topic groups were. The real, positive result
is narrower but still real: genuine external embeddings (a different encoder, real
conversational text) flow cleanly through the whole PN->KC pipeline with no changes needed.

### Run

```
uv run python longmemeval_capacity_probe.py --episodes 20
```

## Phase 22: tuning the fast/slow eta pair

`dual_store_probe.py --sweep` grid-searches fast_eta x slow_eta and reports exact-repeat
recognition, load-after-11-writes, and the same/diff-topic ratio for each pair, instead of
asserting phase 18's first-guess (0.5/0.03) was any good.

**Result:** the same/diff-topic ratio barely moves across the whole grid (1.56-1.65) -- only
the overall magnitude scales with the etas. What *does* move a lot: recognition strength per
unit of capacity damage improves monotonically as fast_eta goes up and slow_eta goes down.
Pushing further past the swept grid:

| fast_eta | slow_eta | exact repeat | load @ 11 | recognition/damage ratio |
| --- | --- | --- | --- | --- |
| 0.50 | 0.03 (phase 18's guess) | -26.5% | -49.6% | 0.53 |
| 0.70 | 0.01 | -35.5% | -49.6% | 0.72 |
| 0.90 | 0.005 | -45.2% | -50.5% | 0.90 |
| 0.99 | 0.001 | -49.6% | -50.2% | **0.99** |

Makes sense once you see why: a fast store with eta near 1 saturates (near-zero) on its very
first hit, so *further* writes to an already-near-zero row barely erode it further --
capacity damage self-limits. Meanwhile a single write still gives it (and so the sum) close
to its maximum possible single-shot depression. The caveat this trades away: at eta near 1
the fast store becomes near-binary (touched vs. not), losing any graded distinction between
"touched once" and "touched repeatedly" within that channel -- the slow store is what would
have to carry that if it mattered.

### Run

```
uv run python dual_store_probe.py --sweep
uv run python dual_store_probe.py --fast-eta 0.99 --slow-eta 0.001
```

## Phase 23: combining tagging/capture with the dual-store, and a metric trap caught mid-experiment

`tagged_dual_store_probe.py` merges phase 20 and phase 18/22 into one mechanism, closer to
the actual biological claim: fast store changes on every event and leaves a decaying tag;
slow store changes ONLY when a later salient event captures whatever is still tagged
(content-free). Question: does an early salient event determine whether an item becomes a
*lasting* memory (protected in the slow store) versus staying transient and vulnerable to
being overwritten by later unrelated writes?

**First result, using erosion as % of each condition's own post-encoding value, looked
backwards:** captured -40.64% vs. uncaptured -23.18% -- capture looked *worse*. Checking the
absolute numbers caught why before trusting it: capture's own extra write (the salient event
itself, plus the capture step) makes its post-encoding baseline about half the size of
uncaptured's (101,438 vs. 210,053) purely as a side effect of capture happening at all, and
the same absolute erosion against a smaller baseline reads as a bigger percentage. That's a
property of the metric, not of the mechanism.

**Corrected, comparing absolute erosion (the fair number here):** captured loses 40,859 to 10
subsequent unrelated writes; uncaptured loses 48,532. Captured is *more* resistant, not less
-- the opposite conclusion from the % framing. Modest (about 16% less absolute erosion), but
the right direction, and worth carrying forward: percentage-of-own-baseline is the wrong
comparison whenever the write itself changes the baseline.

### Run

```
uv run python tagged_dual_store_probe.py
```

## Phase 24: habituation -- write strength scaled by how fresh the KC still is

`memory.write_surprise_gated()`: scale the depression by how undepressed (relative to the
pristine connectome) the currently active KCs still are. A never-touched KC gets the full
`eta`; a KC already heavily depressed by past writes gets a progressively smaller one --
diminishing returns on repeated exposure to the same thing, instead of every write costing
the same regardless of what's left to change. Single store, no tag/capture bookkeeping, no
second store.

**Result -- the cleanest single mechanism found in this line so far:**

- Habituation curve on repeated exposure to the same sentence: **-30.0% -> -21.0% -> -16.6%
  -> -13.8% -> -11.9%** over 5 exposures -- a genuine, monotonic diminishing-returns curve
  (Altmann & Gray 2002: decay rate adapts to update frequency).
- Exact-repeat recognition unchanged from `write()`: first exposure still -30.0% (freshness
  starts at 1.0, so the very first write is identical to the flat rule).
- Same-topic vs. different-topic interference preserved: -7.14% vs -4.47%, 1.60x ratio
  (unbudgeted was 1.57x) -- discrimination survives.
- Load after 11 writes: **-57.6%**, versus unbudgeted's -71.2% -- meaningfully better capacity
  behavior, self-limiting the same way pushing fast_eta near 1 did in phase 22 (an
  already-near-zero row has little left to lose), but from ONE store instead of two.

### Run

```
uv run python surprise_gated_probe.py
```

## Phase 25: habituation + dual-store together -- the best result in this line so far

`habituated_dual_store_probe.py`: each of the fast and slow stores now habituates against
its *own* pristine reference independently (phase 24's rule, applied twice), instead of
picking one mechanism over the other.

**Every text-memory mechanism tried, on the same 12-sentence/4-topic battery:**

| mechanism | exact repeat | load @ 11 writes | same/diff ratio |
| --- | --- | --- | --- |
| unbudgeted (phase 5/7) | -30.0% | -71.2% | 1.57 |
| dual-store, eta 0.5/0.03 (phase 18) | -26.5% | -49.6% | 1.63 |
| dual-store, tuned eta 0.99/0.001 (phase 22) | -49.6% | -50.2% | ~1.5 |
| habituation only (phase 24) | -30.0% | -57.6% | 1.60 |
| **habituation + dual-store (phase 25)** | -26.5% | **-41.5%** | 1.55 |

Best capacity behavior of every mechanism that didn't gut recognition to get there (phase
17's `competitive` mode still wins on raw capacity, -0.7%, but at -0.25% recognition -- see
phase 17). -41.5% at 11 writes, keeping -26.5% single-shot recognition and a same/diff ratio
in the same range as everything else: two independent, cheap ideas (parallel time constants,
diminishing returns per KC) compound instead of trading off against each other.

**One more row, added after phase 32 needed magnitude-weighting for the sound line:**
`memory.write_magnitude_weighted()` (phase 32) on this same text battery gives -15.2%
recognition / -49.6% load / 1.55 ratio -- worse on recognition than phase 25's combination,
no better on load. The circuit-size contrast is the finding: magnitude-weighting was
*essential* for the small, convergent sound circuit (phase 31/32, where every input reaches
the same ~164 downstream neurons and only relative magnitude carries content), but text's
large, sparse KC population already differs in *which* KCs are active from one sentence to
the next, so weighting by magnitude on top of that mostly just throws away recognition
strength for little capacity gain. The right write rule depends on how convergent the
circuit downstream of it actually is.

### Run

```
uv run python habituated_dual_store_probe.py
```

## Phase 27: vision -- interference tracks measured similarity, once labels stop lying about it

Phase 16 found human topic labels (coffee/dog/alarm) didn't predict interference and
diagnosed why: a same-topic pair (cup vs. beans) can look nothing alike in raw contrast/shape
even though a human calls both "coffee." `vision_measured_similarity_probe.py` retests the
actual claim without that confound: rank all 15 pairs among the same 6 real images by their
*measured* KC-set overlap, ignoring topic labels entirely, and compare the single most-similar
pair against the single least-similar pair directly.

**Result:** most-similar pair (`cup` vs. `smoke_detector`, overlap 0.708) interferes -27.01%;
least-similar pair (`golden` vs. `smoke_detector`, overlap 0.330) interferes -20.90%. Right
direction, cleanly. Phase 15/16's "no discrimination" was a mislabeling artifact -- comparing
by an assumed human category that didn't match what the circuit's contrast-based codes
actually treat as similar -- not a failure of the interference mechanism, which tracks real
measured overlap exactly the way phase 6 already showed for text.

### Run

```
uv run python vision_measured_similarity_probe.py
```

## Phase 28: tagging/capture combined with the habituated dual-store

`tagged_habituated_probe.py` folds phase 20's tagging/capture into phase 25's habituated
dual-store (capture itself is now also habituated, scaled by the slow store's own remaining
freshness) -- the fullest mechanism combination in this line, using phase 23's corrected
(absolute, not %-of-own-baseline) erosion metric from the start.

**Result:** captured items still erode less than uncaptured ones (24,479 vs. 27,794 absolute
erosion over 10 unrelated writes) -- the protective direction from phase 23 holds. But the
*relative* size of that protection shrinks once habituation is in the mix (about 11.9% less
erosion here, versus 15.8% in phase 23's non-habituated version) -- habituation already
compresses overall erosion, so there's less room left for capture to additionally protect.
The mechanisms compound, they don't cancel, but they're not simply additive either.

### Run

```
uv run python tagged_habituated_probe.py
```

## Phase 29: the sound line's input side -- no random projection anywhere

Picks up phase 9's fetched AMMC(R)/JO-A/B circuit, untouched since. `audio_to_jo.py`
synthesizes tones, computes their spectral energy via FFT in the 200-350 Hz band Kamikouchi
et al. 2009 identify as JO-A/B's real courtship-song-range tuning, and injects that energy
directly into the real 13 JO-A/B neurons -- no random matrix, unlike the text/vision lines,
because a real physiological tuning curve was already available to use instead. JO-C/E
(gravity/wind-tuned) are deliberately left at zero for a pure tone, rather than inventing a
mapping for a stimulus they don't actually respond to.

**Result:** a 250 Hz tone (inside the band) drives the real downstream AMMC neurons (sum
899.0); a 2000 Hz tone (outside it) drives nothing (0.0); white noise, spreading energy
across the whole spectrum, gives a small but real response (28.3) proportional to the sliver
of its energy that happens to fall in-band. Real transduction, real synapses, correct
qualitative ordering.

### Run

```
uv run python audio_to_jo.py
```

## Phase 30: sound memory -- real recognition, but a hard limit on identity

`audio_memory_probe.py` wires phase 29's input through a KC-equivalent sparse code (same
one-hop-then-top-10% shape as the vision line) and real write/recall against the "other ->
other" AMMC recurrence.

**Result:** 250 Hz repeated shows the same clean recognition signature as text and vision
(-30.0%). But white noise -- a completely different signal -- *also* drops by exactly -30.0%
after writing 250 Hz (41,154 -> 28,808), the identical percentage. Diagnosed, not
hand-waved: `build_stimulus` puts a single scalar (band_energy) on the *same* 13 JO-A/B
neurons for every sound, so every possible stimulus in this design is the same direction
vector, scaled by a different magnitude. `propagate` is linear and `sparse_code`'s top-k
selection is scale-invariant, so **every sound activates the exact same downstream KC set**,
regardless of what it actually is -- only the magnitude differs. Recognition survives (it's
magnitude-driven), but content/identity discrimination is impossible by construction, not
just weak the way vision's was (phase 15/16). A real fix would need genuinely
multi-channel input -- e.g. a filterbank across several frequency bands, each driving a
different subset of JO-A/B (or a larger population), instead of one shared scalar gain.

### Run

```
uv run python audio_memory_probe.py
```

## Phase 31: fixing the input didn't fix it -- write() itself only sees a binary mask

Tried the fix phase 30 proposed: `build_stimulus_multichannel` (`audio_to_jo.py`), a
log-spaced filterbank giving each of the 13 real JO-A/B neurons a different frequency band's
energy. Checked at the continuous-value level it worked -- cosine similarity between very
different sounds' raw propagated drive is as low as 0.0127 (250 Hz vs. 2000 Hz), nothing like
phase 30's exact scalar-multiple problem.

**But `audio_memory_probe.py` still showed all three sounds dropping by exactly -30.0% after
writing one of them.** Diagnosed: only ~164 of 2,631 downstream neurons are reachable at all
from this 13-neuron population -- a real anatomical bottleneck (convergence), not a bug --
so every sound's *binary* active mask (`code > 0`) is identical even though the continuous
values genuinely differ. And `memory.write()` only ever looks at that binary mask
(`weights[active, :] *= 1 - eta`) -- it never uses the code's magnitude at all. When
everything shares the same active set, every write scales the exact same rows by the exact
same factor regardless of which sound triggered it, so recall of *anything* sharing that set
drops by the same percentage. This never showed up in the text/KC or vision/lamina lines
because their populations are large enough that different real inputs naturally select
different active sets -- it's a real limit of `write()`'s design (magnitude-blind), just one
that only a small, convergent circuit like this one exposes.

### Run

```
uv run python audio_memory_probe.py  # now uses build_stimulus_multichannel; see the printed cosine table
```

## Phase 32: magnitude-weighted write -- real content discrimination, finally

`memory.write_magnitude_weighted()`: scale each active row's depression by its own value
relative to the strongest active row, instead of `write()`'s binary-mask-only rule -- close
in spirit to `write_competitive`'s normalized shares (phase 17), but for content instead of
budget.

**Result, same three sounds, `audio_memory_probe.py` switched to this write rule:** 250 Hz
exact repeat -17.4% (recognition survives, just less total depression than the old rule --
expected, since a magnitude-weighted rule spends less budget on rows that were only weakly
active); 2000 Hz after writing 250 Hz -0.3% (cosine similarity to it was only 0.0127 --
almost no interference, correctly); white noise after writing 250 Hz -3.5% (cosine similarity
0.3435 -- moderate interference, correctly in between). The ordering of interference now
tracks the ordering of real cosine similarity exactly. This is the sound line's write
mechanism finally seeing content, not just occupancy -- the fix phases 29-31 were building
toward.

### Run

```
uv run python audio_memory_probe.py
```

## Phase 33: one more verified odorant, and why the list stays short on purpose

Checked candidates by web search rather than from memory, per this project's own rule
against guessing citations. Added: ethyl butyrate -> DM2 (Hallem & Carlson 2006, Or22a --
the same glomerulus ethanol already used, a second real keyword onto real `DM2_lPN`
neurons). Rejected as a candidate: isoamyl acetate (banana smell) -- real literature says it
activates *six* different glomeruli (VC3, DM3, VM5v, DA4m, DC1, VA4), not one, so it doesn't
fit this table's "one specific, well-established pairing" design; forcing it to a single
glomerulus would be exactly the kind of unverified simplification this line has tried to
avoid.

### Run

```
uv run python odor_embedder.py --text "The cocktail smelled faintly of ethyl butyrate."
```

## Phase 34: opponent-channel valence -- the mechanism works, the labels don't exist yet

`valence_probe.py` implements RESEARCH.md's licensed shape: MBON output split into two
opponent channels, readout is their *difference* (not a raw sum), and reinforcement
depresses the channel signaling the OPPOSITE valence (Owald et al. 2015; Aso et al. 2014) --
never potentiates the agreeing one.

**Honesty check done before running anything:** this project has no verified mapping from
our fetched MBON type names (`MBON14`, `MBON06`, etc.) to which real compartments are
appetitive vs. aversive (Aso et al. 2014 assigns that per real compartment; we haven't
checked our numeric labels against it). So the two channels here are an *arbitrary* 50/50
split of the 75 real MBON neurons -- this tests whether the opponent-channel mechanism
behaves correctly given a sign, not a claim about which specific real neurons carry which
real valence.

**Result:** baseline valence score +7,649. Positive/appetitive reinforcement shifts it to
+30,128 (+22,479); negative/aversive reinforcement shifts it to -17,124 (-24,773).
Symmetric, correctly signed, using real KC->MBON weights throughout.

### Run

```
uv run python valence_probe.py
```

## Phase 35: real question-type labels for the LongMemEval same/different test

Phase 21 grouped LongMemEval candidates by episode index and found no discrimination,
diagnosing that the 8 distractor sessions per episode aren't a topic cluster. Checked
whether a real label exists instead of giving up on external validation: `longmemeval.py`
(the `neural-memory` project) already threads a genuine `question_type` category
(`single-session-user`, `multi-session`, `knowledge-update`, `temporal-reasoning`, etc.)
through to the exact `longmemeval_deferred.pt` file phase 21 read -- as a `question_type`
int array plus `question_type_names` -- just not consulted there.

**Result:** `longmemeval_question_type_probe.py`, 60 real episodes sampled across
knowledge-update/multi-session/temporal-reasoning: same-type -1.39% (n=79,776) vs.
different-type -1.38% (n=150,144) -- indistinguishable, again. Diagnosed the same way as
phase 16/27's vision finding: `question_type` is a real external label, but it's a *task-
structure* category (how the question is built), not a *content-topic* one -- two
"multi-session" questions can be about completely unrelated things. The pattern holds across
both lines now: no label (assumed category, episode index, task type) substitutes for
*measured* similarity (phase 6, phase 27) -- only actually measuring it does.

### Run

```
uv run python longmemeval_question_type_probe.py --episodes 60
```

## Phase 36: the mechanism comparison was ranking the wrong number, and it inverts

Phase 25 picked habituated dual-store as this line's best mechanism on a table of two columns:
"exact repeat" and "load @ 11 writes". Both are measured against the *pristine* weights, so the
load number contains the item's own write. Decomposing the same table:

| mechanism | exact repeat | load @ 11 | interference alone |
| --- | ---: | ---: | ---: |
| unbudgeted | -30.0% | -71.2% | -41.2% |
| dual-store 0.5/0.03 | -26.5% | -49.6% | -23.1% |
| habituation only | -30.0% | -57.6% | -27.6% |
| habituation + dual (phase 25's pick) | -26.5% | -41.5% | -15.0% |
| **dual-store 0.99/0.001 (phase 22, called worse)** | -49.6% | -50.2% | **-0.6%** |

A mechanism that writes shallowly scores well on "load" for that reason alone. The row phase 22
set aside erodes twenty-five times less from unrelated writes than the row phase 25 adopted.

Neither column answers what a memory is for. Both are read on items that *were* written; nothing
in that table involves an item the circuit has never seen, so it cannot separate "the memory
survived" from "everything sank less".

`recognition_under_load_probe.py` asks the missing question. Write a random half of a document
set, then read every document, written and unwritten alike, and score how separable the two
groups are (AUC; 0.5 is no separation). Run on 512 real LongMemEval document embeddings, 20
trials per load:

| written | unbudgeted | **saturating** | dual | dual_tuned | habituation | habituated_dual | stored_codes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.9023 | **1.0000** | 0.9383 | 1.0000 | 0.9273 | 0.9555 | 1.0000 |
| 16 | 0.8023 | **1.0000** | 0.8275 | 1.0000 | 0.8100 | 0.8221 | 1.0000 |
| 32 | 0.7372 | **0.9836** | 0.7131 | 0.9683 | 0.7142 | 0.7015 | 0.9999 |
| 64 | 0.6591 | **0.8684** | 0.5998 | 0.7956 | 0.6218 | 0.5980 | 1.0000 |
| 128 | 0.6055 | **0.7409** | 0.5424 | 0.6140 | 0.5636 | 0.5414 | 0.9993 |
| 256 | 0.5831 | **0.6349** | 0.5284 | 0.5319 | 0.5433 | 0.5252 | 0.9986 |

The ranking inverts. Phase 25's habituated dual-store is the **worst** mechanism at every load
above 16, reaching 0.5252 at 256 writes, which is chance. What wins is the simplest rule in the
set: a single store with `eta` near 1, which nearly zeroes each written row. `saturating` is not
even in phase 25's table -- it is what is left of the tuned dual-store once the slow store is
removed, and removing it *helps* at every load, so the "dual" framing was carrying nothing. The
mechanism is a saturating, effectively binary mark on the KC rows that were active.

Phase 25's two ideas were not wrong about their own metric; the metric rewarded writing less,
and writing less is exactly what loses recognition under load.

**A capacity number for this circuit, which the line did not have before.** With 2,053 KC and 75
MBON, the best mechanism separates written from unwritten at 0.98 AUC for 32 documents, 0.87 for
64, 0.74 for 128, and 0.63 for 256. The `stored_codes` column is the upper bound: keep one KC
code per written item and score by best overlap. It stays at 1.0 throughout, so the KC codes
themselves are perfectly separable and every bit of the degradation above is the fixed-size
synaptic state, not the encoding. That column is not a competitor -- its memory grows with the
load while the synaptic state does not -- but it does say where the loss lives.

### Run

```
uv run python recognition_under_load_probe.py --trials 20 --loads 8,16,32,64,128,256
```

## Phase 37: the capacity follows from two integers, and correlation is what bends it

If phase 36's winning rule is a near-binary mark on the KC rows that were active, then the
capacity is not a curve to be measured mechanism by mechanism -- it is a Bloom filter's
false-positive rate. An unwritten document reads as written only if *every* cell it activates was
already marked:

    p   = 1 - (1 - k/m)^N        each cell marked by at least one of N writes
    AUC = 1 - 0.5 * p^k          a false positive is a tie, worth 0.5

Nothing is fitted. `m` = 2,053 Kenyon cells is counted from the connectome, `k` is counted from
the codes, `N` is the load. `bloom_capacity_probe.py` runs it against three stores: the real KC
codes with the counting rule phase 36 used, the same codes clamped so a row is marked once and
never further, and codes shuffled to independent random subsets of the same size.

| keep | k | written | predicted | real, counting | real, binary | shuffled, binary |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 41 | 128 | 0.9801 | 0.7109 | 0.7438 | **0.9821** |
| 0.02 | 41 | 512 | 0.5007 | 0.5739 | 0.5958 | **0.4986** |
| 0.05 | 103 | 128 | 0.5661 | 0.7503 | 0.7794 | **0.5687** |
| 0.10 | 205 | 128 | 0.5001 | 0.7568 | 0.7687 | **0.5120** |
| 0.10 | 205 | 32 | 0.9996 | 0.9576 | 0.9705 | **0.9985** |
| 0.20 | 411 | 32 | 0.6381 | 0.9634 | 0.9672 | **0.6351** |
| 0.40 | 821 | 32 | 0.5000 | 0.8764 | 0.8875 | **0.5069** |

**The formula is essentially exact for independent codes.** Across a twentyfold range of `k` and
a sixtyfourfold range of `N`, the shuffled column sits within noise of a prediction with no free
parameters: 0.9801 vs 0.9821, 0.5661 vs 0.5687, 0.6381 vs 0.6351, 0.5007 vs 0.4986. This memory
is a Bloom filter over Kenyon cell identities, and that is now demonstrated rather than asserted.

**Counting adds nothing.** Clamping so a row is marked once rather than multiplied down again on
every later write changes the result by at most 0.03 and usually less. The `eta`-near-1 store is
already binary in effect, which is why the single saturating rule beat every graded mechanism in
phase 36.

**Correlation is the whole deviation, and it cuts both ways.** Real codes come from one shared
projection of one corpus, so they overlap far more than random subsets of the same size, and the
gap runs in opposite directions depending on the regime:

- Where the filter has room (`k` small, moderate load), correlation **hurts**: at `k`=41 and 128
  writes, 0.7438 against a predicted 0.9801. Correlated documents mark the same popular cells,
  so a new document finds its cells already taken more often than chance says.
- Where the filter is saturated, correlation **helps**: at `k`=205 and 128 writes the prediction
  and the shuffled control both collapse to chance, 0.5001 and 0.5120, while real codes still
  read 0.7687. Writes concentrating on popular cells is exactly what leaves the rest of the
  population unmarked, so saturation arrives far later than a uniform filter would predict.

Correlated codes are therefore not simply a degraded version of independent ones. They trade peak
capacity for a much softer ceiling, which is the more useful shape for a memory that will be
pushed past its design load.

**The sparsity optimum, and where the fly sits.** Sweeping `keep_ratio` on real codes (binary
store):

| written | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.7438 | **0.7794** | 0.7687 | 0.7014 | 0.6019 |
| 512 | **0.5958** | 0.5852 | 0.5699 | 0.5469 | 0.5186 |

The optimum is 0.02-0.05, and `RESEARCH.md` records the real APL-enforced sparsening as keeping
**5 to 10 percent** of Kenyon cells active. The lower edge of the biological range is where the
measured capacity peaks. Two cautions before reading anything into that: these codes come from a
random projection of text embeddings rather than real odour responses, so the correlation
structure driving the optimum is the corpus's and not the fly's; and this line has been running
at `keep_ratio` 0.10 throughout, which the table shows is past the peak at every load measured.

### Run

```
uv run python bloom_capacity_probe.py --trials 20 --keep-ratios 0.02,0.05,0.10,0.20,0.40 --loads 8,32,128,512
```

## Not built yet

- Text/memory line: phase 34's opponent-channel valence used an arbitrary MBON split; a real
  one would need verifying our fetched MBON type names against Aso et al. 2014's real
  per-compartment valence assignments, not attempted. A genuine content-topic label for
  LongMemEval (phase 35 found `question_type` isn't one) would need something else -- maybe
  clustering by embedding similarity itself, which is circular for validating this system.
- Vision line: still unresolved whether to keep using real downloaded images or add a
  text-to-image step (which would relocate the "borrowed meaning" problem, not solve it --
  see phase 11), and propagation past the lamina into the medulla for R7/R8's real pathway.
- Only 7 odorants are covered, deliberately (phase 33) -- most real odorants are
  broadly-tuned across several glomeruli, not a single clean pairing.
- Sound line: real content discrimination now works (phase 32); still unresolved whether 13
  sound-tuned neurons is enough for anything beyond simple tones/noise, or whether AMMC(L)
  needs pulling in too; and real recorded sound (courtship song, environmental audio) instead
  of only synthetic tones/noise.

## Phase 38: the wiring is not for capacity, and the valence task I built could not test it either

FlyGM (arXiv 2602.17997) instantiated the whole fly connectome as a graph controller and beat three
baselines on locomotion. Its sharpest control is a degree-preserving rewiring: every neuron keeps its
exact in- and out-degree, only its partners change, so losing to it means the advantage is *who* is
wired to whom rather than how much wiring there is. The paper never tests memory — searching its text
for "mushroom body" returns nothing, because walking does not need one, even though those neurons sit
in the graph it uses. This asks the question it left.

### Linear associative capacity: the connectome carries no prior

`wiring_prior_probe.py` stores associations in the real KC-to-MBON support and compares against a
curveball rewiring and a uniform shuffle of the same synapses.

| items | real | rewired | random | real over rewired | real over random |
|---:|---:|---:|---:|---|---|
| 500 | 0.9776 | 0.9750 | 0.9976 | +0.0026 NOT resolved | **-0.0200 resolved** |
| 1000 | 0.7874 | 0.7898 | 0.8760 | -0.0023 NOT resolved | **-0.0886 resolved** |
| 2000 | 0.4050 | 0.4053 | 0.4822 | -0.0003 NOT resolved | **-0.0772 resolved** |

By the falsification fixed in advance, this fails: the real wiring does not beat the degree-preserving
rewiring, so on this measure its structure is worth no more than its degree sequence. And the degree
sequence itself costs capacity — a uniform shuffle of the same 30,543 synapses wins everywhere.

That agrees with the literature rather than overturning it. Random expansion is what maximises coding
capacity, and the mushroom body pays capacity for selectivity, biasing connectivity toward what
matters to the animal. **Capacity is not what this wiring is for.**

### What it is for, read straight off the anatomy

`compartment_valence.py` counts PAM against PPL1 presynapses per compartment and takes the sign their
ratio implies. Nothing is fitted and no behavioural data is used.

| reward | | punishment | |
|---|---:|---|---:|
| b'2 | +1.000 (PAM 12659 : PPL 1) | a3 | -1.000 (PAM 0 : PPL 2424) |
| g5 | +0.996 | CA | -1.000 |
| b2 | +0.996 | a2 | -0.999 |
| b1 | +0.988 | a'3 | -0.993 |
| g4 | +0.987 | a'2 | -0.969 |
| g3 | +0.907 | g1 | -0.856 |
| b'1 | +0.750 | a'1 | -0.800 |
| a1 | +0.727 | g2 | -0.448 |

The textbook split — horizontal lobes reward, vertical lobes punish — falls out of raw synapse counts.

### The valence task, and why it does not answer the question

`valence_store_probe.py` stores one bit per pattern under the fly's depression rule and reads the
valence-weighted MBON ensemble vote.

| items | real | rewired | random | real over rewired |
|---:|---:|---:|---:|---|
| 400 | 0.9427 | 0.9397 | 0.9403 | +0.0030 resolved |
| 2000 | 0.7216 | 0.7192 | 0.7224 | +0.0023 NOT resolved |
| 8000 | 0.6331 | 0.6318 | 0.6334 | +0.0013 resolved |

The real wiring does beat the rewiring, consistently in sign and resolved at two loads. But the effect
is one to three tenths of a percent, and no load separates it from a uniform shuffle.

**The design is the reason, and it is mine.** The signs are indexed by MBON column, and rewiring
shuffles which Kenyon cell reaches which MBON while leaving that column-to-sign map intact. Valence
therefore survives any rewiring, which is why all three conditions agree to three decimals. The task
injects the compartment structure into the scoring rather than requiring the wiring to supply it, so
it cannot test the wiring.

### Two defects, both caught by their own numbers

The first capacity probe scored exactly chance for every wiring: the docstring described a delta rule
the code did not implement, so codes were pushed through a fixed matrix with nothing stored. Printing
chance alongside the scores is what made it obvious.

The first rewiring permuted column indices globally and summed collisions, losing 5,255 of 30,543
synapses and breaking row degree outright. "The real wiring beats a rewiring" then only meant it beat
a sparser matrix, and the +0.0495 it produced at 1000 items became -0.0023 once the curveball trade
held both degree sequences and the value multiset exactly.

### What to measure next

Before designing another task, measure whether there is block structure to exploit: do Kenyon cells
partition by the compartment their MBONs sit in, more than a degree-preserving rewiring would give?
If they do not, no task will separate these conditions and the valence result above is the ceiling.

## Phase 39: there is block structure, and it does not protect a lesson

Phase 38 ended with a question a task could not answer: is there any structure in KC-to-MBON for a
task to exploit? `compartment_structure_probe.py` measures the wiring directly, no learning involved.

| wiring | top-compartment share | compartments reached |
|---|---:|---:|
| real | 0.2642 | **7.31** |
| rewired | 0.2898 | 9.90 |
| random | 0.2999 | 9.37 |
| even spread | 0.0625 | 16 |

`reach: real over rewired` is **-2.59 [-2.59, -2.58] resolved**. A real Kenyon cell touches 2.6 fewer
compartments than a rewiring holding every degree fixed. The structure is there, and it is selective:
concentration is *lower* in the real wiring, so a cell does not favour one compartment — it picks a
few and splits between them, which is what an axon running along one lobe and synapsing where it
passes would produce.

### The prediction that followed, and failed

If a lesson lands in fewer compartments, two lessons should collide less.
`interference_probe.py` stores lesson A, stores lesson B on top, and asks how much of A survives —
as a ratio against each wiring's own A-alone score, so a wiring that holds less cannot look robust by
having less to lose.

| wiring | A alone | A after B | kept |
|---|---:|---:|---:|
| real | 0.9985 | 0.9521 | 0.9535 |
| rewired | 0.9977 | 0.9556 | 0.9578 |
| random | 1.0000 | 0.9871 | **0.9871** |

`kept: real over rewired` is -0.0044 [-0.0116, +0.0033], **not resolved**. `kept: real over random` is
-0.0336 [-0.0382, -0.0293], resolved against the real wiring. By the falsification fixed in advance,
reaching fewer compartments buys no protection.

The reasoning was wrong about where interference happens. Two lessons collide when the same Kenyon
cells are active for both, and that is decided by the sparse code, not by the wiring; the wiring only
decides which MBONs the collision reaches. Compartment selectivity is real and irrelevant to this.

### Three angles, one answer

| task | real vs rewired | real vs random |
|---|---|---|
| linear capacity | no difference | **random wins** |
| signed valence | real wins (design fault: signs indexed by MBON column) | no difference |
| interference | no difference | **random wins** |

Within linear associative storage the connectome's structure earns nothing, asked three different
ways. That is not "the connectome is useless" — FlyGM wins against this same control on closed-loop
sensorimotor control. It says this frame is the wrong one. The mushroom body was shaped by natural
odour statistics, metabolic cost, developmental constraint, and above all by choosing actions; only
storage was measured here.

`compartment_valence.py` is what survives as a standalone result: valence signs read off raw anatomy,
nothing fitted, recovering the textbook horizontal-reward / vertical-punishment split.
