"""A learned selector for what to put in front of the model.

Phase 70 measured behaviour for the first time and found the shape that
justifies this project: injecting memory helps (+0.1030 over none), choosing it
by similarity does not beat taking the most recent thing (-0.0481, unresolved),
and choosing it the way an oracle would is worth +0.3835 more than similarity.
So there is something to select and cosine does not find it.

The oracle picks the memory items most like the action about to be taken, which
needs the future and cannot be run. But it is a supervised target: learn a
scorer that, seeing only the state so far, ranks what the oracle would have
picked. Training never touches the generator, which would be far too slow; the
generator appears only at evaluation, where the question is whether this
selector's four items beat cosine's four.

Held out by trajectory, so no stream trained on is ever scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ActionSelector(nn.Module):
    """Bilinear scorer over memory, starting exactly at cosine similarity."""

    def __init__(self, dim: int, rank: int = 64, context: int = 4) -> None:
        super().__init__()
        self.left = nn.Parameter(torch.zeros(dim, rank))
        self.right = nn.Parameter(torch.zeros(dim, rank))
        nn.init.normal_(self.left, std=0.01)
        weights = torch.zeros(context)
        weights[0] = 8.0
        self.context_logits = nn.Parameter(weights)
        self.context = context

    def cue(self, recent: Tensor) -> Tensor:
        """Blend the newest few steps; more may be handed in than are weighted."""
        taken = recent[: self.context]
        if len(taken) == 1:
            return taken[0]
        weights = F.softmax(self.context_logits[: len(taken)], dim=0)
        return F.normalize(weights @ taken, dim=0)

    def forward(self, memory: Tensor, recent: Tensor) -> Tensor:
        cue = self.cue(recent)
        return memory @ (cue + self.right @ (self.left.T @ cue))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("artifacts/open_swe_tool_stream.pt")
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--recent", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--save", type=Path, default=Path("artifacts/action_selector.pt"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    turns = F.normalize(payload["turns"], dim=-1)
    offsets = payload["offsets"]
    streams = len(offsets) - 1

    rows = []
    best_state = None
    for seed in (int(value) for value in args.seeds.split(",")):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(streams, generator=generator)
        cut = int(streams * (1.0 - args.holdout))
        train_streams, eval_streams = order[:cut].tolist(), order[cut:].tolist()
        model = ActionSelector(turns.shape[-1], args.rank, args.context)
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        def sample(pool: list[int]) -> tuple[Tensor, Tensor, int] | None:
            index = pool[int(torch.randint(len(pool), (1,), generator=generator))]
            start, stop = int(offsets[index]), int(offsets[index + 1])
            if stop - start < args.warmup + args.recent + 2:
                return None
            position = int(
                torch.randint(args.warmup, stop - start - 1, (1,), generator=generator)
            ) + start
            memory = turns[start : position - args.recent]
            if len(memory) < 2:
                return None
            recent = turns[position - args.recent : position].flip(0)
            # What the oracle would have chosen, which needs the action being
            # predicted and is therefore a label rather than a usable rule.
            target = int((memory @ turns[position]).argmax())
            return memory, recent, target

        @torch.no_grad()
        def evaluate(pool: list[int] | None = None) -> dict[str, float]:
            """Agreement with the oracle. Run on the training pool too, because
            a target that cannot be fitted there is not learnable at all, which
            is a different problem from overfitting."""
            model.eval()
            agree_trained, agree_cosine = [], []
            for _ in range(2000):
                drawn = sample(pool if pool is not None else eval_streams)
                if drawn is None:
                    continue
                memory, recent, target = drawn
                agree_trained.append(float(int(model(memory, recent).argmax()) == target))
                agree_cosine.append(float(int((memory @ recent[0]).argmax()) == target))
            model.train()
            return {
                "trained_agreement": sum(agree_trained) / max(len(agree_trained), 1),
                "cosine_agreement": sum(agree_cosine) / max(len(agree_cosine), 1),
                "positions": len(agree_trained),
            }

        before = evaluate()
        for _ in range(args.steps):
            drawn = sample(train_streams)
            if drawn is None:
                continue
            memory, recent, target = drawn
            loss = F.cross_entropy(
                model(memory, recent).unsqueeze(0) / 0.05,
                torch.tensor([target]),
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        after = evaluate()
        seen = evaluate(train_streams)
        rows.append({"seed": seed, "before": before, "after": after, "seen": seen})
        if best_state is None:
            best_state = {key: value.clone() for key, value in model.state_dict().items()}

    print(f"{streams} streams, {len(rows)} seeds, held out by stream\n")
    print(f"{'reader':>20} {'agrees with the oracle':>24}")
    average = lambda key, which: sum(row[which][key] for row in rows) / len(rows)
    print(f"{'cosine':>20} {average('cosine_agreement', 'after'):24.4f}")
    print(f"{'trained, before':>20} {average('trained_agreement', 'before'):24.4f}")
    print(f"{'trained, after':>20} {average('trained_agreement', 'after'):24.4f}")
    print(f"{'trained, on seen':>20} {average('trained_agreement', 'seen'):24.4f}")

    if args.save and best_state is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state": best_state, "rank": args.rank, "context": args.context}, args.save
        )
        print(f"\nsaved selector to {args.save}")

    output = {"config": vars(args), "per_seed": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
