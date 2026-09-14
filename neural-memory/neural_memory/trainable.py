from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def hard_sparse(projected: Tensor, active: int) -> Tensor:
    """Keep the strongest units and renormalise, letting gradient reach them.

    Which units win is a discrete choice and carries no gradient, but the values
    at the winners do, which is enough to move the projection that produced them.
    """
    if not 1 <= active <= projected.shape[-1]:
        raise ValueError("active must lie between one and the unit count")
    mask = torch.zeros_like(projected)
    mask.scatter_(-1, projected.abs().topk(active, dim=-1).indices, 1.0)
    return F.normalize(projected * mask, dim=-1)


class TrainableMemory(nn.Module):
    """The assembled store with its projections learned rather than random.

    Everything the redesign kept is still here: a sparse expanded code whose
    width follows the state's own magnitude, and delta-rule storage with no
    decay. What changes is that the two projections are now trained against the
    only thing the store is asked to do, which is return each document's value
    ahead of every other stored one.
    """

    def __init__(
        self,
        feature_dim: int,
        key_dim: int = 512,
        value_dim: int = 64,
        base_active: float = 6.1,
        active_exponent: float = 1.4,
        lowest_active: int = 8,
        highest_active: int = 96,
        adaptive: bool = True,
        fixed_active: int = 32,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or key_dim < 1 or value_dim < 1:
            raise ValueError("dimensions must be positive")
        self.key_projection = nn.Linear(feature_dim, key_dim, bias=False)
        self.value_projection = nn.Linear(feature_dim, value_dim, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(2.0))
        self.base_active = base_active
        self.active_exponent = active_exponent
        self.lowest_active = lowest_active
        self.highest_active = highest_active
        self.adaptive = adaptive
        self.fixed_active = fixed_active

    def _active(self, matrix: Tensor) -> int:
        if not self.adaptive:
            return self.fixed_active
        # The width is a discrete choice read off the state, so it is detached:
        # the projection is trained through the values at the chosen units, not
        # through how many of them there are.
        size = max(float(matrix.detach().norm()), 1.0)
        wanted = round(self.base_active * size**self.active_exponent)
        return int(min(self.highest_active, max(self.lowest_active, wanted)))

    def forward(self, block: Tensor) -> Tensor:
        """Store a set of documents and score every probe against every value."""
        if block.ndim != 2:
            raise ValueError("block must have shape [documents, features]")
        projected = torch.tanh(self.key_projection(block))
        values = F.normalize(torch.tanh(self.value_projection(block)), dim=-1)
        matrix = values.new_zeros(values.shape[-1], projected.shape[-1])
        keys = []
        for row, value in zip(projected, values, strict=True):
            key = hard_sparse(row, self._active(matrix))
            keys.append(key)
            matrix = matrix + torch.outer(value - matrix @ key, key)
        reads = F.normalize(torch.stack(keys) @ matrix.T, dim=-1)
        return reads @ values.T * self.log_scale.exp().clamp(max=100.0)

    @torch.no_grad()
    def discrimination(self, block: Tensor) -> float:
        """Fraction of documents whose probe returns their own value first."""
        logits = self.forward(block)
        return float((logits.argmax(dim=-1) == torch.arange(len(block))).float().mean())
