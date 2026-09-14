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


def soft_sparse(projected: Tensor, threshold: Tensor, temperature: Tensor) -> Tensor:
    """Sparsify by a graded gate rather than a counted budget.

    A top-k selection is a discrete choice, so nothing about how many units it
    keeps can be learned through it. Gating each unit on how far its magnitude
    exceeds a threshold makes the effective width a continuous quantity, and the
    threshold that sets it becomes trainable.
    """
    gate = torch.sigmoid((projected.abs() - threshold) / temperature.clamp(min=1e-3))
    return F.normalize(projected * gate, dim=-1), gate


class WidthPolicy(nn.Module):
    """A sampled, discrete width whose parameters still receive a gradient.

    Top-k stays exactly as it was, so the sparsity keeps the discreteness that
    makes it worth having. What changes is how the count is chosen: a width is
    drawn from a distribution whose mean follows the state's magnitude, and the
    distribution's parameters are trained by policy gradient against the episode's
    own loss. The forward pass never sees a relaxation.
    """

    def __init__(
        self,
        base: float = 6.1,
        exponent: float = 1.4,
        sigma: float = 3.0,
        lowest: int = 8,
        highest: int = 96,
    ) -> None:
        super().__init__()
        if not 1 <= lowest <= highest:
            raise ValueError("bounds must satisfy 1 <= lowest <= highest")
        self.log_base = nn.Parameter(torch.tensor(base).log())
        self.exponent = nn.Parameter(torch.tensor(exponent))
        self.log_sigma = nn.Parameter(torch.tensor(sigma).log())
        self.lowest = lowest
        self.highest = highest

    def mean_width(self, size: float) -> Tensor:
        return self.log_base.exp() * max(size, 1.0) ** self.exponent

    def sample(self, size: float) -> tuple[int, Tensor]:
        """Draw a width and return it with the log-probability of the draw."""
        mean = self.mean_width(size)
        sigma = self.log_sigma.exp().clamp(min=0.5)
        distribution = torch.distributions.Normal(mean, sigma)
        drawn = distribution.sample()
        log_probability = distribution.log_prob(drawn)
        width = int(min(self.highest, max(self.lowest, round(float(drawn)))))
        return width, log_probability


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
        learned_width: bool = False,
        sampled_width: bool = False,
        full_components: bool = False,
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
        self.learned_width = learned_width
        self.sampled_width = sampled_width
        self.width_policy = WidthPolicy(
            base_active, active_exponent, lowest=lowest_active, highest=highest_active
        )
        # The gate's threshold rises with the log of the state's magnitude, which
        # is the same signal the hand-fitted rule used; here its offset and slope
        # are learned instead of being fitted to two points of a sweep.
        # Initialised against the projection's own scale: |proj| reaches about
        # 0.125 and the top twenty of five hundred and twelve units sit above
        # 0.061, so the gate starts near the width the hand-fitted rule chose.
        self.width_offset = nn.Parameter(torch.tensor(0.061))
        self.width_slope = nn.Parameter(torch.tensor(-0.012))
        self.width_temperature = nn.Parameter(torch.tensor(0.004))
        # The components the Phase 29 ablation removed by hand, restored as
        # trainable quantities so the optimiser can decide their size instead.
        # All start where that phase set them.
        self.full_components = full_components
        self.slow_logit = nn.Parameter(torch.tensor(0.25).logit())
        self.slow_decay_logit = nn.Parameter(torch.tensor(0.99).logit())
        self.tag_decay_logit = nn.Parameter(torch.tensor(0.9).logit())
        self.capture_logit = nn.Parameter(torch.tensor(0.25).logit())

    def _active(self, matrix: Tensor) -> int:
        if not self.adaptive:
            return self.fixed_active
        # The width is a discrete choice read off the state, so it is detached:
        # the projection is trained through the values at the chosen units, not
        # through how many of them there are.
        size = max(float(matrix.detach().norm()), 1.0)
        wanted = round(self.base_active * size**self.active_exponent)
        return int(min(self.highest_active, max(self.lowest_active, wanted)))

    def forward(self, block: Tensor, weak_first: float = 1.0) -> Tensor:
        """Store a set of documents and score every probe against every value.

        With the full component set the first document is written weakly, leaving
        a tag, and a reinforcement addressed at it follows the writes. That gives
        the tag decay and the capture rate something to act on, so the optimiser
        can size them rather than have them sized by hand.
        """
        if block.ndim != 2:
            raise ValueError("block must have shape [documents, features]")
        projected = torch.tanh(self.key_projection(block))
        values = F.normalize(torch.tanh(self.value_projection(block)), dim=-1)
        matrix = values.new_zeros(values.shape[-1], projected.shape[-1])
        slow = values.new_zeros(values.shape[-1], projected.shape[-1])
        keys = []
        tags: list[Tensor] = []
        self._last_widths: list[float] = []
        self._last_log_probabilities: list[Tensor] = []
        for index, (row, value) in enumerate(zip(projected, values, strict=True)):
            if self.sampled_width:
                width, log_probability = self.width_policy.sample(
                    float(matrix.detach().norm())
                )
                self._last_log_probabilities.append(log_probability)
                key = hard_sparse(row, width)
                self._last_widths.append(float(width))
            elif self.learned_width:
                size = torch.tensor(max(float(matrix.detach().norm()), 1.0))
                threshold = self.width_offset + self.width_slope * size.log()
                key, gate = soft_sparse(row, threshold, self.width_temperature)
                self._last_widths.append(float(gate.detach().sum()))
            else:
                key = hard_sparse(row, self._active(matrix))
                self._last_widths.append(float((key != 0).sum()))
            keys.append(key)
            full = torch.outer(value - matrix @ key, key)
            strength = weak_first if (self.full_components and index == 0) else 1.0
            matrix = matrix + strength * full
            if self.full_components:
                tag_decay = self.tag_decay_logit.sigmoid()
                tags = [tag * tag_decay for tag in tags]
                if strength < 1.0:
                    tags.append((1.0 - strength) * full)
                slow = self.slow_decay_logit.sigmoid() * slow + (
                    self.slow_logit.sigmoid() * torch.outer(value - slow @ key, key)
                )
        if self.full_components and tags:
            stacked = torch.stack(tags)
            response = torch.einsum('tij,j->ti', stacked, keys[0])
            current = response.norm(dim=1)
            gain = self.capture_logit.sigmoid() * current * (1.0 - current)
            matrix = matrix + torch.einsum('t,tij->ij', gain, stacked)
        state = matrix + slow if self.full_components else matrix
        reads = F.normalize(torch.stack(keys) @ state.T, dim=-1)
        return reads @ values.T * self.log_scale.exp().clamp(max=100.0)

    @torch.no_grad()
    def discrimination(self, block: Tensor, weak_first: float = 1.0) -> float:
        """Fraction of documents whose probe returns their own value first."""
        logits = self.forward(block, weak_first)
        return float((logits.argmax(dim=-1) == torch.arange(len(block))).float().mean())
