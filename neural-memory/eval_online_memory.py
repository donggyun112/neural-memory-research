"""A memory that learns on the stream it is in, not one trained somewhere else.

Every read in this project so far was fitted on one corpus, frozen, and carried
elsewhere. Phase 54 showed that carrying fails on a different agent, and the
right reading of that is not that the read is bad but that the question was
wrong. A memory is not a pre-learned system. It adapts while it runs, on the
material it is actually seeing.

So there is no training corpus here and nothing to transfer. Each stream gets
its own read, starting at plain cosine, and walks forward: at every position the
read is scored *before* it is updated, so nothing it is graded on has touched
its weights. The baseline is the same hard cosine pick, static, on the identical
positions.

The comparison is therefore not "does a learned transform generalise" but "does
adapting online beat not adapting", which is the property a memory is supposed
to have.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class OnlineRead(nn.Module):
    """A low-rank shift on the cue, starting at the identity."""

    def __init__(self, dim: int, rank: int = 16, temperature: float = 0.02) -> None:
        super().__init__()
        self.left = nn.Parameter(torch.zeros(dim, rank))
        self.right = nn.Parameter(torch.zeros(dim, rank))
        nn.init.normal_(self.left, std=0.01)
        self.temperature = temperature

    def scores(self, memory: Tensor, cue: Tensor) -> Tensor:
        return memory @ (cue + self.right @ (self.left.T @ cue))

    def forward(self, memory: Tensor, cue: Tensor) -> Tensor:
        weights = F.softmax(self.scores(memory, cue) / self.temperature, dim=0)
        return F.normalize(weights @ memory, dim=0)

    @torch.no_grad()
    def constrain(self, budget: float) -> None:
        """Keep the shift inside a ball around the identity.

        Phase 56 left a residual that lowering the rate does not touch, which
        points at total distance travelled rather than step size. How far the
        shift can move is bounded by the product of its factors' norms, so
        rescaling both when that product exceeds a budget bounds the distance
        directly. A memory that does not drift is one with a limit on how far it
        can get from what it started as.
        """
        if budget <= 0:
            return
        size = float(self.left.norm() * self.right.norm())
        if size > budget:
            scale = (budget / size) ** 0.5
            self.left.mul_(scale)
            self.right.mul_(scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--foils", type=int, default=99)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch", type=int, default=1, help="positions averaged per update")
    parser.add_argument("--replay", type=int, default=0, help="earlier positions added per update")
    parser.add_argument(
        "--budget", type=float, default=0.0, help="cap on how far the shift may travel; 0 is free"
    )
    parser.add_argument("--max-positions", type=int, default=400, help="per stream")
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="skip streams shorter than this, so every bin holds the same streams",
    )
    parser.add_argument(
        "--memory-window",
        type=int,
        default=0,
        help="hold the memory to the most recent N items; 0 keeps everything",
    )
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    turns = F.normalize(payload["turns"], dim=-1)
    offsets = payload["offsets"]
    streams = len(offsets) - 1

    futures = torch.zeros_like(turns)
    valid = torch.zeros(len(turns), dtype=torch.bool)
    for index in range(streams):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        for position in range(start, stop - args.horizon):
            futures[position] = turns[position + 1 : position + 1 + args.horizon].mean(0)
            valid[position] = True
    futures = F.normalize(futures, dim=-1)
    pool = torch.nonzero(valid).flatten()

    rows = []
    for seed in (int(value) for value in args.seeds.split(",")):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        # A memory starts empty and should be expected to do nothing early. An
        # average over all positions mixes that stretch with one where the read
        # has had hundreds of updates, so the position each score came from is
        # kept and the comparison is reported as a curve.
        online_hits, static_hits, steps_taken = [], [], []
        ceiling_hits, blend_hits, moved, travelled = [], [], [], []
        # One frozen copy for the whole run: the untrained blend is the same
        # function everywhere, so it needs no per-stream state.
        frozen = OnlineRead(turns.shape[-1], args.rank)
        for index in range(streams):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            usable = range(start + args.warmup, stop - args.horizon)
            # Short streams appear in the early bins and not the late ones, so a
            # curve over bins compares different populations and confounds "more
            # updates" with "different stream". Requiring every stream to reach
            # the last bin fixes the population across the whole curve.
            if len(usable) < args.min_length:
                continue
            # A fresh read per stream: nothing is carried in from anywhere.
            model = OnlineRead(turns.shape[-1], args.rank)
            # Decay pulls the shift toward zero, and zero is the identity here,
            # so it is an anchor to plain cosine rather than generic shrinkage.
            # Without it the read drifts: it gains for the first fifty updates
            # and then walks away from anything useful.
            optimiser = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            seen: list[int] = []
            pending: list[int] = []
            for step, position in enumerate(usable):
                if step >= args.max_positions:
                    break
                # Updates and memory size grow together, so an expiry could be
                # either. Holding the window fixed makes the task the same
                # difficulty throughout and leaves only the update count moving.
                low = max(start, position - args.memory_window) if args.memory_window else start
                memory = turns[low:position]
                future = futures[position]
                foils = pool[torch.randperm(len(pool), generator=generator)[: args.foils]]
                candidates = torch.cat([future.unsqueeze(0), futures[foils]])
                with torch.no_grad():
                    # Scored before the update, so this position never trained it.
                    read = model(memory, turns[position])
                    scored = candidates @ read
                    online_hits.append(float((scored > scored[0]).sum() == 0))
                    fixed = memory[int((memory @ turns[position]).argmax())]
                    scored = candidates @ fixed
                    static_hits.append(float((scored > scored[0]).sum() == 0))
                    # The best item actually in memory, chosen with the future in
                    # hand. Headroom is this minus the static pick, and it is the
                    # quantity the gain is supposed to track.
                    scored = candidates @ memory[int((memory @ future).argmax())]
                    ceiling_hits.append(float((scored > scored[0]).sum() == 0))
                    # The same read at initialisation, never updated. It blends
                    # where the static pick takes one item, so without it
                    # "online adaptation helps" cannot be separated from
                    # "blending helps" -- and a longer horizon makes the target
                    # an average of more actions, which favours a blend on its
                    # own.
                    scored = candidates @ frozen(memory, turns[position])
                    blend_hits.append(float((scored > scored[0]).sum() == 0))
                    # An effect of exactly zero is not a small effect. With the
                    # softmax this sharp, a small shift changes scores without
                    # changing which item dominates, and then the read is frozen
                    # in output even while its parameters move. This counts how
                    # often the adapted read actually attends somewhere else.
                    moved.append(
                        float(
                            int(model.scores(memory, turns[position]).argmax())
                            != int(frozen.scores(memory, turns[position]).argmax())
                        )
                    )
                steps_taken.append(step)
                seen.append(position)
                pending.append(position)
                # Phase 55 found a single-position gradient noisy enough that
                # continued updating walks the read somewhere useless. Batching
                # averages several before stepping; replay adds earlier positions
                # so the update is not only about the newest thing. Both draw
                # solely from the past, so the causal guarantee is unchanged.
                if len(pending) >= args.batch:
                    chosen = list(pending)
                    if args.replay and len(seen) > len(pending):
                        older = seen[: -len(pending)]
                        picks = torch.randint(
                            len(older), (min(args.replay, len(older)),), generator=generator
                        )
                        chosen += [older[int(pick)] for pick in picks]
                    losses = []
                    for past in chosen:
                        past_low = (
                            max(start, past - args.memory_window)
                            if args.memory_window
                            else start
                        )
                        past_memory = turns[past_low:past]
                        if not len(past_memory):
                            continue
                        past_foils = pool[
                            torch.randperm(len(pool), generator=generator)[: args.foils]
                        ]
                        past_candidates = torch.cat(
                            [futures[past].unsqueeze(0), futures[past_foils]]
                        )
                        losses.append(
                            F.cross_entropy(
                                (past_candidates @ model(past_memory, turns[past])).unsqueeze(0)
                                / 0.05,
                                torch.zeros(1, dtype=torch.long),
                            )
                        )
                    if losses:
                        optimiser.zero_grad(set_to_none=True)
                        torch.stack(losses).mean().backward()
                        optimiser.step()
                        model.constrain(args.budget)
                    pending.clear()
            travelled.append(float(model.left.norm() * model.right.norm()))
        rows.append(
            {
                "seed": seed,
                "positions": len(online_hits),
                "online": float(np.mean(online_hits)),
                "static": float(np.mean(static_hits)),
                "_online": online_hits,
                "_static": static_hits,
                "_steps": steps_taken,
                "ceiling": float(np.mean(ceiling_hits)),
                "blend": float(np.mean(blend_hits)),
                "attention_moved": float(np.mean(moved)),
                "shift_norm": float(np.mean(travelled)),
                "_blend": blend_hits,
            }
        )

    online = np.concatenate([np.array(row["_online"]) for row in rows])
    static = np.concatenate([np.array(row["_static"]) for row in rows])
    gap = online - static
    generator = np.random.default_rng(7)
    draws = [gap[generator.integers(0, len(gap), len(gap))].mean() for _ in range(5000)]
    low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    print(f"{streams} streams, {rows[0]['positions']} scored positions per seed, "
          f"{len(rows)} seeds, no pre-training and nothing transferred\n")
    print(f"{'reader':>26} {'top-1':>9}")
    ceiling = float(np.mean([row["ceiling"] for row in rows]))
    blend = np.concatenate([np.array(row["_blend"]) for row in rows])
    print(f"{'static hard pick':>26} {static.mean():9.4f}")
    print(f"{'untrained blend':>26} {blend.mean():9.4f}")
    print(f"{'online, adapting as it goes':>26} {online.mean():9.4f}")
    print(f"{'best single item (ceiling)':>26} {ceiling:9.4f}")
    print(f"{'headroom':>26} {ceiling - static.mean():9.4f}")
    print(
        f"{'attention moved':>26} "
        f"{float(np.mean([row['attention_moved'] for row in rows])):9.4f}"
    )
    print(
        f"{'shift norm at stream end':>26} "
        f"{float(np.mean([row['shift_norm'] for row in rows])):9.4f}"
    )
    # The decomposition the horizon sweep demands: how much of the gain is
    # blending rather than adapting.
    for label, left, right in (
        ("blend minus static", blend, static),
        ("online minus blend", online, blend),
    ):
        difference = left - right
        maker = np.random.default_rng(7)
        draws = [
            difference[maker.integers(0, len(difference), len(difference))].mean()
            for _ in range(3000)
        ]
        edge_low, edge_high = (
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        )
        print(
            f"{label:>26} {difference.mean():+9.4f}  [{edge_low:+.4f}, {edge_high:+.4f}]"
            f" {'resolved' if edge_low > 0 or edge_high < 0 else 'no'}"
        )
    print(
        f"\nonline minus static: {gap.mean():+.4f}  [{low:+.4f}, {high:+.4f}]"
        f" {'resolved' if low > 0 or high < 0 else 'NOT resolved'}"
    )

    steps = np.concatenate([np.array(row["_steps"]) for row in rows])
    print(f"\n{'updates seen':>14} {'positions':>10} {'static':>8} {'online':>8} {'difference':>11}")
    curve = {}
    edges = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, 10**6)]
    for low_edge, high_edge in edges:
        chosen = (steps >= low_edge) & (steps < high_edge)
        if not chosen.any():
            continue
        binned = online[chosen] - static[chosen]
        difference = float(binned.mean())
        # The early bins are the claim, so they get an interval rather than a
        # point estimate: whether a memory gains anything before it has learned
        # much is the question, and it is answered on a few thousand positions.
        maker = np.random.default_rng(7)
        resampled = [
            binned[maker.integers(0, len(binned), len(binned))].mean() for _ in range(3000)
        ]
        bin_low = float(np.percentile(resampled, 2.5))
        bin_high = float(np.percentile(resampled, 97.5))
        curve[f"{low_edge}-{high_edge}"] = {
            "positions": int(chosen.sum()),
            "static": float(static[chosen].mean()),
            "online": float(online[chosen].mean()),
            "difference": difference,
            "low": bin_low,
            "high": bin_high,
        }
        label = f"{low_edge}-{high_edge}" if high_edge < 10**6 else f"{low_edge}+"
        print(
            f"{label:>14} {int(chosen.sum()):10d} {static[chosen].mean():8.4f}"
            f" {online[chosen].mean():8.4f} {difference:+11.4f}"
            f"  [{bin_low:+.4f}, {bin_high:+.4f}]"
            f" {'resolved' if bin_low > 0 or bin_high < 0 else 'no'}"
        )

    for row in rows:
        row.pop("_online", None)
        row.pop("_static", None)
        row.pop("_steps", None)
        row.pop("_blend", None)
    output = {
        "config": vars(args),
        "per_seed": rows,
        "online": float(online.mean()),
        "static": float(static.mean()),
        "difference": {"mean": float(gap.mean()), "low": low, "high": high},
        "curve": curve,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
