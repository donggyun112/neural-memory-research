"""Train the read against the future, and see how much of the gap it closes.

Phase 44 found a target that exists at every position -- what the memory should
surface at time t is whatever matters at t+1 -- and measured a gap of about
0.133 between plain association and the best item actually in memory. This asks
whether a learned read captures any of it.

The scoring function is bilinear, `score(q, m) = m^T (I + UV^T) q`, initialised
at the identity so training starts exactly at the cosine baseline. Anything it
gains is attributable to learning rather than to a different parameterisation.
Selection is a softmax over memory, so the whole read is differentiable, and the
loss is InfoNCE: the softly selected vector has to pick the real future out of
the same foils the evaluation uses.

Questions are split by haystack, so no turn from an evaluated conversation is
ever trained on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PredictiveRead(nn.Module):
    """A bilinear read that begins as plain cosine and can move away from it."""

    def __init__(self, dim: int, rank: int = 32) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.left = nn.Parameter(torch.zeros(dim, rank))
        self.right = nn.Parameter(torch.zeros(dim, rank))
        # Cosines span a narrow range, so a temperature of one leaves the softmax
        # nearly uniform and the read becomes the mean of memory rather than a
        # selection. Starting sharp makes the untrained model equal the cosine
        # baseline it is supposed to begin at.
        self.log_temperature = nn.Parameter(torch.tensor(0.02).log())
        nn.init.normal_(self.left, std=0.01)

    def scores(self, memory: Tensor, cue: Tensor) -> Tensor:
        # I + UV^T, so at initialisation (V = 0) this is exactly memory @ cue.
        shifted = cue + self.right @ (self.left.T @ cue)
        return memory @ shifted

    def forward(self, memory: Tensor, cue: Tensor) -> Tensor:
        weights = F.softmax(self.scores(memory, cue) / self.log_temperature.exp(), dim=0)
        return F.normalize(weights @ memory, dim=0)


def episode_positions(
    offsets: Tensor, index: int, warmup: int, horizon: int
) -> range:
    start, stop = int(offsets[index]), int(offsets[index + 1])
    return range(start + warmup, stop - horizon)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/longmemeval_turns.pt")
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--foils", type=int, default=99)
    parser.add_argument("--distant", type=int, default=64)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-positions", type=int, default=40)
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    turns = F.normalize(payload["turns"], dim=-1)
    offsets = payload["offsets"]
    episodes = len(offsets) - 1

    futures = torch.zeros_like(turns)
    valid = torch.zeros(len(turns), dtype=torch.bool)
    for index in range(episodes):
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
        order = torch.randperm(episodes, generator=generator)
        cut = int(episodes * (1.0 - args.holdout))
        train_episodes, eval_episodes = order[:cut].tolist(), order[cut:].tolist()

        model = PredictiveRead(turns.shape[-1], args.rank)

        @torch.no_grad()
        def evaluate() -> dict[str, float]:
            model.eval()
            hits = {name: [] for name in ("trained", "cosine", "oracle", "recency")}
            old_hits = {name: [] for name in hits}
            evaluation = torch.Generator().manual_seed(seed + 1)
            for index in eval_episodes:
                start = int(offsets[index])
                usable = list(episode_positions(offsets, index, args.warmup, args.horizon))
                if not usable:
                    continue
                picked = torch.randperm(len(usable), generator=evaluation)[
                    : args.eval_positions
                ]
                for slot in picked.tolist():
                    position = usable[slot]
                    memory = turns[start:position]
                    future = futures[position]
                    foils = pool[torch.randperm(len(pool), generator=evaluation)[: args.foils]]
                    candidates = torch.cat([future.unsqueeze(0), futures[foils]])
                    best = int((memory @ future).argmax())
                    reads = {
                        "trained": model(memory, turns[position]),
                        "cosine": memory[int((memory @ turns[position]).argmax())],
                        "oracle": memory[best],
                        "recency": memory[-1],
                    }
                    old = (len(memory) - 1 - best) >= args.distant
                    for name, read in reads.items():
                        scored = candidates @ read
                        hit = float((scored > scored[0]).sum() == 0)
                        hits[name].append(hit)
                        if old:
                            old_hits[name].append(hit)
            return {
                **{f"{name}_top1": sum(v) / len(v) for name, v in hits.items()},
                **{
                    f"{name}_top1_old": (sum(v) / len(v)) if v else float("nan")
                    for name, v in old_hits.items()
                },
                # Every reader saw the same positions, so a difference between two
                # of them is paired and resolves far tighter than a spread over
                # five seeds suggests.
                "_old_hits": {name: list(v) for name, v in old_hits.items()},
            }

        before = evaluate()
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        model.train()
        training = torch.Generator().manual_seed(seed + 2)
        for _ in range(args.steps):
            index = train_episodes[int(torch.randint(len(train_episodes), (1,), generator=training))]
            start = int(offsets[index])
            usable = list(episode_positions(offsets, index, args.warmup, args.horizon))
            if not usable:
                continue
            position = usable[int(torch.randint(len(usable), (1,), generator=training))]
            memory = turns[start:position]
            foils = pool[torch.randperm(len(pool), generator=training)[: args.foils]]
            candidates = torch.cat([futures[position].unsqueeze(0), futures[foils]])
            read = model(memory, turns[position])
            loss = F.cross_entropy(
                (candidates @ read).unsqueeze(0) / 0.05, torch.zeros(1, dtype=torch.long)
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        after = evaluate()
        rows.append({"seed": seed, "before": before, "after": after})

    print(f"{episodes} conversations, {len(rows)} seeds, held-out by conversation\n")
    print(f"{'reader':>18} {'top-1':>9} {'top-1, old':>12} {'per-seed spread, old':>22}")
    summary = {}
    # The soft read differs from plain cosine in two ways at once: it blends
    # several memory items instead of taking one, and it is trained. Reporting
    # the untrained blend separately is what separates the parameterisation from
    # the learning -- the distinction this project has got wrong twice.
    for name, key, label in (
        ("recency", "before", "recency"),
        ("cosine", "before", "hard pick, cosine"),
        ("trained", "before", "soft blend, untrained"),
        ("trained", "after", "soft blend, trained"),
        ("oracle", "before", "best single item"),
    ):
        values = [row[key][f"{name}_top1_old"] for row in rows]
        cells = {
            "top1": sum(row[key][f"{name}_top1"] for row in rows) / len(rows),
            "top1_old": sum(values) / len(values),
            "spread": max(values) - min(values),
        }
        summary[label] = cells
        print(
            f"{label:>18} {cells['top1']:9.4f} {cells['top1_old']:12.4f}"
            f" {cells['spread']:22.4f}"
        )
    import numpy as np

    paired = {
        label: np.concatenate([np.array(row[key]["_old_hits"][name]) for row in rows])
        for name, key, label in (
            ("recency", "before", "recency"),
            ("cosine", "before", "hard"),
            ("trained", "before", "blend"),
            ("trained", "after", "trained"),
            ("oracle", "before", "oracle"),
        )
    }
    print(f"\n{'comparison, old positions':>34} {'difference':>11} {'95% interval':>22}")
    comparisons = {}
    for left, right in (
        ("trained", "blend"),
        ("blend", "hard"),
        ("hard", "recency"),
        ("trained", "hard"),
    ):
        gap = paired[left] - paired[right]
        generator = np.random.default_rng(7)
        draws = [gap[generator.integers(0, len(gap), len(gap))].mean() for _ in range(5000)]
        low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        comparisons[f"{left} minus {right}"] = {
            "difference": float(gap.mean()),
            "low": low,
            "high": high,
        }
        print(
            f"{left + ' minus ' + right:>34} {gap.mean():+11.4f}  [{low:+.4f}, {high:+.4f}]"
            f" {'resolved' if low > 0 or high < 0 else 'NOT resolved'}"
        )
    print(f"\npaired positions compared: {len(paired['trained'])}")

    for row in rows:
        for key in ("before", "after"):
            row[key].pop("_old_hits", None)
    output = {
        "config": vars(args),
        "per_seed": rows,
        "summary": summary,
        "paired": comparisons,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
