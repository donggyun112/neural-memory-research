from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class MemoryConfig:
    """Every component the redesign validated, each switchable for ablation."""

    key_dim: int = 512
    value_dim: int = 64
    base_active: float = 6.1
    active_exponent: float = 1.4
    lowest_active: int = 8
    highest_active: int = 96
    fixed_active: int = 32
    adaptive_sparsity: bool = True
    parallel_stores: bool = True
    slow_strength: float = 0.25
    slow_decay: float = 0.99
    fast_decay: float = 1.0
    tags: bool = True
    tag_decay: float = 0.9
    capture: float = 0.25


@dataclass
class MemoryState:
    fast: Tensor
    slow: Tensor
    eligibility: list[Tensor] = field(default_factory=list)
    active_counts: list[int] = field(default_factory=list)


def sparsify(projected: Tensor, active: int) -> Tensor:
    kept = torch.zeros_like(projected)
    indices = projected.abs().topk(active).indices
    kept[indices] = projected[indices]
    return F.normalize(kept, dim=0)


class AssembledMemory:
    """The redesign's parts in one store, so they can be ablated together.

    Sparsity follows the state's own magnitude (Phase 28). Nothing decays in the
    fast store, because error-correcting storage already handles the interference
    decay would answer (Phase 27). A second slower store makes two answers to one
    cue representable and lets which one is expressed change with delay (Phase
    24). Writes leave fading tags so a later event can rescue a weak trace inside
    a window (Phase 22), and that event acts through coincidence and a prediction
    error rather than by reading content (Phase 23).
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()

    def empty(self) -> MemoryState:
        shape = (self.config.value_dim, self.config.key_dim)
        return MemoryState(fast=torch.zeros(shape), slow=torch.zeros(shape))

    def _active(self, state: MemoryState) -> int:
        config = self.config
        if not config.adaptive_sparsity:
            return config.fixed_active
        size = max(float(state.fast.norm()), 1.0)
        wanted = round(config.base_active * size**config.active_exponent)
        return int(min(config.highest_active, max(config.lowest_active, wanted)))

    def write(self, state: MemoryState, projected: Tensor, value: Tensor, strength: float = 1.0) -> Tensor:
        """Store one document and return the key it was given."""
        config = self.config
        key = sparsify(projected, self._active(state))
        full = torch.outer(value - state.fast @ key, key)
        state.fast = config.fast_decay * state.fast + strength * full
        if config.tags:
            state.eligibility = [tag * config.tag_decay for tag in state.eligibility]
            if strength < 1.0:
                state.eligibility.append((1.0 - strength) * full)
        if config.parallel_stores:
            state.slow = config.slow_decay * state.slow + config.slow_strength * torch.outer(
                value - state.slow @ key, key
            )
        state.active_counts.append(int((key != 0).sum()))
        return key

    def reinforce(self, state: MemoryState, probe: Tensor, sign: float) -> None:
        """Let a later event modulate what is still eligible, without reading it."""
        config = self.config
        if not config.tags or not state.eligibility:
            return
        tags = torch.stack(state.eligibility)
        response = torch.einsum("tij,j->ti", tags, probe)
        current = response.norm(dim=1)
        wanted = torch.full_like(current, 1.0 if sign > 0 else 0.0)
        gain = config.capture * current.clamp(min=0.0) * (wanted - current)
        state.fast = state.fast + torch.einsum("t,tij->ij", gain, tags)

    def replay(self, state: MemoryState, selected: Tensor, rate: float = 0.15) -> None:
        if not state.eligibility:
            return
        tags = torch.stack(state.eligibility)
        state.fast = state.fast + rate * torch.einsum("t,tij->ij", selected, tags)

    def read(self, state: MemoryState, key: Tensor) -> Tensor:
        combined = state.fast @ key
        if self.config.parallel_stores:
            combined = combined + state.slow @ key
        return combined

    def discriminate(self, state: MemoryState, keys: Tensor, values: Tensor) -> Tensor:
        """Whether each probe returns its own value ahead of every other stored one."""
        reads = F.normalize(torch.stack([self.read(state, key) for key in keys]), dim=1)
        agreement = reads @ F.normalize(values, dim=1).T
        own = agreement.diagonal()
        rivals = agreement.masked_fill(
            torch.eye(len(values), dtype=torch.bool), -torch.inf
        ).max(dim=1).values
        return (own > rivals).float()
