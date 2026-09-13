from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


RecallMode = Literal["candidate", "query", "joint"]


@dataclass(frozen=True)
class ContextTraceState:
    traces: Tensor


@dataclass(frozen=True)
class SelectiveTraceState:
    traces: Tensor
    gate_logits: Tensor
    selected: Tensor
    strengths: Tensor


@dataclass(frozen=True)
class DeferredTraceState:
    traces: Tensor
    write_logits: Tensor
    provisional: Tensor
    consolidation_logits: Tensor
    selected: Tensor
    strengths: Tensor


def hard_top_k(logits: Tensor, eligible: Tensor, ratio: float) -> tuple[Tensor, Tensor]:
    """Select a fixed fraction of the eligible slots and keep a gradient path.

    The forward pass is a genuine discrete choice, so a rejected slot is absent
    rather than downweighted.  The straight-through strength carries the later
    loss back into whichever gate produced ``logits``.
    """
    if logits.shape != eligible.shape:
        raise ValueError("logits and eligible mask must have the same shape")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be inside (0, 1]")
    masked = logits.masked_fill(~eligible, -torch.inf)
    selected = torch.zeros_like(eligible)
    for row in range(len(eligible)):
        available = int(eligible[row].sum())
        if available == 0:
            continue
        keep = max(1, round(available * ratio))
        selected[row, masked[row].topk(keep).indices] = True
    probabilities = masked.sigmoid().masked_fill(~eligible, 0.0)
    strengths = selected.float() + probabilities - probabilities.detach()
    return selected, strengths


@dataclass(frozen=True)
class FastWeightState:
    matrix: Tensor
    gate_logits: Tensor
    selected: Tensor
    surprise: Tensor
    retention: Tensor


class MultiTraceRecallMemory(nn.Module):
    """Store compact candidate traces, then activate them from a later natural query."""

    def __init__(self, feature_dim: int, memory_dim: int = 64) -> None:
        super().__init__()
        if feature_dim < 1 or memory_dim < 1:
            raise ValueError("dimensions must be positive")
        self.candidate_projection = nn.Linear(feature_dim, memory_dim)
        self.query_projection = nn.Linear(feature_dim, memory_dim)
        self.candidate_head = nn.Linear(memory_dim, 1)
        self.query_head = nn.Linear(memory_dim, 1)
        self.relation = nn.Sequential(
            nn.Linear(memory_dim * 4, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, 1),
        )

    def observe(self, candidate_features: Tensor) -> ContextTraceState:
        if candidate_features.ndim != 3:
            raise ValueError("candidates must have shape [batch, traces, feature_dim]")
        return ContextTraceState(torch.tanh(self.candidate_projection(candidate_features)))

    def activate(
        self,
        state: ContextTraceState,
        query_features: Tensor,
        *,
        mode: RecallMode = "joint",
    ) -> Tensor:
        if query_features.ndim != 2:
            raise ValueError("queries must have shape [batch, feature_dim]")
        query = torch.tanh(self.query_projection(query_features))
        expanded = query[:, None, :].expand_as(state.traces)
        if mode == "candidate":
            return self.candidate_head(state.traces).squeeze(-1)
        if mode == "query":
            return self.query_head(expanded).squeeze(-1)
        if mode != "joint":
            raise ValueError(f"unknown mode: {mode}")
        relation = torch.cat(
            (
                state.traces,
                expanded,
                state.traces * expanded,
                (state.traces - expanded).abs(),
            ),
            dim=-1,
        )
        return (
            self.candidate_head(state.traces)
            + self.query_head(expanded)
            + self.relation(relation)
        ).squeeze(-1)

    def alignment_logits(self, state: ContextTraceState, query_features: Tensor) -> Tensor:
        """Auxiliary cosine logits that explicitly align traces with their query."""
        query = torch.tanh(self.query_projection(query_features))
        traces = nn.functional.normalize(state.traces, dim=-1)
        query = nn.functional.normalize(query, dim=-1)
        return (traces * query[:, None, :]).sum(dim=-1) * 10.0

    def forward(
        self,
        candidate_features: Tensor,
        query_features: Tensor,
        *,
        mode: RecallMode = "joint",
    ) -> Tensor:
        return self.activate(self.observe(candidate_features), query_features, mode=mode)


class DualEncoderRecallMemory(nn.Module):
    """Learn a compact cosine space for candidate/query activation."""

    def __init__(self, feature_dim: int, memory_dim: int = 64) -> None:
        super().__init__()
        if feature_dim < 1 or memory_dim < 1:
            raise ValueError("dimensions must be positive")
        self.candidate_projection = nn.Sequential(
            nn.Linear(feature_dim, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, memory_dim),
            nn.LayerNorm(memory_dim),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(feature_dim, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, memory_dim),
            nn.LayerNorm(memory_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def observe(self, candidate_features: Tensor) -> ContextTraceState:
        if candidate_features.ndim != 3:
            raise ValueError("candidates must have shape [batch, traces, feature_dim]")
        traces = nn.functional.normalize(self.candidate_projection(candidate_features), dim=-1)
        return ContextTraceState(traces)

    def activate(
        self,
        state: ContextTraceState,
        query_features: Tensor,
        *,
        mode: RecallMode = "joint",
    ) -> Tensor:
        if query_features.ndim != 2:
            raise ValueError("queries must have shape [batch, feature_dim]")
        if mode != "joint":
            raise ValueError("dual encoder supports only joint candidate/query activation")
        query = nn.functional.normalize(self.query_projection(query_features), dim=-1)
        cosine = (state.traces * query[:, None, :]).sum(dim=-1)
        return cosine * self.logit_scale.exp().clamp(max=100.0) + self.bias

    def forward(
        self,
        candidate_features: Tensor,
        query_features: Tensor,
        *,
        mode: RecallMode = "joint",
    ) -> Tensor:
        return self.activate(self.observe(candidate_features), query_features, mode=mode)


class SelectiveWriteRecallMemory(nn.Module):
    """Choose a fixed-capacity trace set now, then recall it from later context."""

    def __init__(
        self,
        feature_dim: int,
        memory_dim: int = 64,
        keep_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or memory_dim < 1:
            raise ValueError("dimensions must be positive")
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be inside (0, 1]")
        self.keep_ratio = keep_ratio
        self.candidate_projection = nn.Linear(feature_dim, memory_dim)
        self.write_projection = nn.Linear(feature_dim, memory_dim)
        self.recall_projection = nn.Linear(feature_dim, memory_dim)
        self.write_gate = nn.Sequential(
            nn.Linear(memory_dim * 4, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, 1),
        )
        self.recall_head = nn.Sequential(
            nn.Linear(memory_dim * 4, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, 1),
        )

    @staticmethod
    def _relation(left: Tensor, right: Tensor) -> Tensor:
        return torch.cat((left, right, left * right, (left - right).abs()), dim=-1)

    def write(
        self,
        candidate_features: Tensor,
        write_context: Tensor,
        masks: Tensor,
    ) -> SelectiveTraceState:
        if candidate_features.ndim != 3 or write_context.ndim != 2:
            raise ValueError("candidates and write context must be rank 3 and 2")
        if masks.shape != candidate_features.shape[:2]:
            raise ValueError("masks must match candidate trace axes")
        traces = torch.tanh(self.candidate_projection(candidate_features))
        context = torch.tanh(self.write_projection(write_context))[:, None, :]
        context = context.expand_as(traces)
        gate_logits = self.write_gate(self._relation(traces, context)).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~masks, -torch.inf)
        selected = torch.zeros_like(masks)
        for row in range(len(masks)):
            valid = int(masks[row].sum())
            keep = max(1, round(valid * self.keep_ratio))
            indices = gate_logits[row].topk(keep).indices
            selected[row, indices] = True
        probabilities = gate_logits.sigmoid().masked_fill(~masks, 0.0)
        straight_through = selected.float() + probabilities - probabilities.detach()
        stored = traces * straight_through[..., None]
        return SelectiveTraceState(stored, gate_logits, selected, straight_through)

    def recall(self, state: SelectiveTraceState, recall_context: Tensor) -> Tensor:
        if recall_context.ndim != 2:
            raise ValueError("recall context must have shape [batch, feature_dim]")
        context = torch.tanh(self.recall_projection(recall_context))[:, None, :]
        context = context.expand_as(state.traces)
        logits = self.recall_head(self._relation(state.traces, context)).squeeze(-1)
        # An unselected trace is absent at inference, while the straight-through
        # strength carries the final-loss gradient back into the write gate.
        return logits + (state.strengths - 1.0) * 8.0

    def forward(
        self,
        candidate_features: Tensor,
        write_context: Tensor,
        recall_context: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, SelectiveTraceState]:
        state = self.write(candidate_features, write_context, masks)
        return self.recall(state, recall_context), state


class DeferredConsolidationMemory(nn.Module):
    """Hold a provisional set, let a later related event consolidate it, then recall.

    Stage one sees only the candidates: no query and no outcome exist yet, which
    is where a single-stage writer must already commit.  Stage two receives one
    later event and may narrow the provisional set, but cannot recover anything
    stage one dropped.  The gap between this and a single-stage writer at the
    same final capacity is the value of deferring.
    """

    def __init__(
        self,
        feature_dim: int,
        memory_dim: int = 64,
        provisional_ratio: float = 0.5,
        keep_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or memory_dim < 1:
            raise ValueError("dimensions must be positive")
        if not 0.0 < keep_ratio <= provisional_ratio <= 1.0:
            raise ValueError("ratios must satisfy 0 < keep_ratio <= provisional_ratio <= 1")
        self.provisional_ratio = provisional_ratio
        self.keep_ratio = keep_ratio
        self.candidate_projection = nn.Linear(feature_dim, memory_dim)
        self.event_projection = nn.Linear(feature_dim, memory_dim)
        self.recall_projection = nn.Linear(feature_dim, memory_dim)
        self.write_gate = nn.Sequential(
            nn.Linear(memory_dim, memory_dim), nn.GELU(), nn.Linear(memory_dim, 1)
        )
        self.consolidation_head = nn.Sequential(
            nn.Linear(memory_dim * 4, memory_dim), nn.GELU(), nn.Linear(memory_dim, 1)
        )
        self.recall_head = nn.Sequential(
            nn.Linear(memory_dim * 4, memory_dim), nn.GELU(), nn.Linear(memory_dim, 1)
        )

    @staticmethod
    def _relation(left: Tensor, right: Tensor) -> Tensor:
        return torch.cat((left, right, left * right, (left - right).abs()), dim=-1)

    def _paired(self, traces: Tensor, context: Tensor, projection: nn.Module) -> Tensor:
        projected = torch.tanh(projection(context))[:, None, :].expand_as(traces)
        return self._relation(traces, projected)

    def write(self, candidate_features: Tensor, masks: Tensor) -> DeferredTraceState:
        if candidate_features.ndim != 3:
            raise ValueError("candidates must have shape [batch, traces, features]")
        if masks.shape != candidate_features.shape[:2]:
            raise ValueError("masks must match candidate trace axes")
        traces = torch.tanh(self.candidate_projection(candidate_features))
        write_logits = self.write_gate(traces).squeeze(-1)
        provisional, strengths = hard_top_k(write_logits, masks, self.provisional_ratio)
        return DeferredTraceState(
            traces=traces * strengths[..., None],
            write_logits=write_logits,
            provisional=provisional,
            consolidation_logits=torch.zeros_like(write_logits),
            selected=provisional,
            strengths=strengths,
        )

    def consolidate(self, state: DeferredTraceState, event: Tensor) -> DeferredTraceState:
        """Narrow the provisional set using one later event, keeping the discard final."""
        if event.ndim != 2:
            raise ValueError("event must have shape [batch, feature_dim]")
        logits = self.consolidation_head(
            self._paired(state.traces, event, self.event_projection)
        ).squeeze(-1)
        ratio = self.keep_ratio / self.provisional_ratio
        selected, strengths = hard_top_k(logits, state.provisional, ratio)
        combined = state.strengths * strengths
        return DeferredTraceState(
            traces=state.traces * strengths[..., None],
            write_logits=state.write_logits,
            provisional=state.provisional,
            consolidation_logits=logits,
            selected=selected,
            strengths=combined,
        )

    def recall(self, state: DeferredTraceState, query: Tensor) -> Tensor:
        logits = self.recall_head(
            self._paired(state.traces, query, self.recall_projection)
        ).squeeze(-1)
        return logits + (state.strengths - 1.0) * 8.0

    def forward(
        self,
        candidate_features: Tensor,
        event: Tensor,
        query: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, DeferredTraceState]:
        state = self.consolidate(self.write(candidate_features, masks), event)
        return self.recall(state, query), state


class FastWeightRecallMemory(nn.Module):
    """Write selected inputs into a bounded parametric associative-memory state."""

    def __init__(
        self,
        feature_dim: int,
        memory_dim: int = 64,
        keep_ratio: float = 0.5,
        adaptive_retention: bool = False,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or memory_dim < 1:
            raise ValueError("dimensions must be positive")
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be inside (0, 1]")
        self.keep_ratio = keep_ratio
        self.memory_dim = memory_dim
        self.adaptive_retention = adaptive_retention
        self.candidate_key = nn.Linear(feature_dim, memory_dim)
        self.candidate_value = nn.Linear(feature_dim, memory_dim)
        self.query_key = nn.Linear(feature_dim, memory_dim)
        self.write_gate = nn.Sequential(
            nn.Linear(memory_dim * 4, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, 1),
        )
        self.retention_logit = nn.Parameter(torch.tensor(4.6))
        self.retention_gate: nn.Sequential | None = None
        if adaptive_retention:
            self.retention_gate = nn.Sequential(
                nn.Linear(memory_dim * 4, memory_dim),
                nn.GELU(),
                nn.Linear(memory_dim, 1),
            )
            nn.init.zeros_(self.retention_gate[-1].weight)
            nn.init.constant_(self.retention_gate[-1].bias, 4.6)
        self.retention_logit.requires_grad_(not adaptive_retention)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def _views(self, candidate_features: Tensor) -> tuple[Tensor, Tensor]:
        keys = nn.functional.normalize(torch.tanh(self.candidate_key(candidate_features)), dim=-1)
        values = nn.functional.normalize(
            torch.tanh(self.candidate_value(candidate_features)), dim=-1
        )
        return keys, values

    @staticmethod
    def _read(matrix: Tensor, key: Tensor) -> Tensor:
        return torch.bmm(matrix, key[..., None]).squeeze(-1)

    def _update(
        self,
        matrix: Tensor,
        key: Tensor,
        value: Tensor,
        strength: Tensor,
        retention: Tensor,
    ) -> Tensor:
        prediction = self._read(matrix, key)
        error = value - prediction
        delta = error[..., :, None] * key[..., None, :]
        return retention[..., None, None] * matrix + strength[..., None, None] * delta

    def _retention(self, relation: Tensor) -> Tensor:
        if self.adaptive_retention:
            assert self.retention_gate is not None
            return self.retention_gate(relation).squeeze(-1).sigmoid()
        return self.retention_logit.sigmoid().expand(len(relation))

    def write(self, candidate_features: Tensor, masks: Tensor) -> FastWeightState:
        if candidate_features.ndim != 3:
            raise ValueError("candidates must have shape [batch, traces, feature_dim]")
        if masks.shape != candidate_features.shape[:2]:
            raise ValueError("masks must match candidate trace axes")
        keys, values = self._views(candidate_features)
        matrix = candidate_features.new_zeros(
            len(candidate_features), self.memory_dim, self.memory_dim
        )
        logits: list[Tensor] = []
        surprises: list[Tensor] = []
        for slot in range(candidate_features.shape[1]):
            prediction = self._read(matrix, keys[:, slot])
            error = values[:, slot] - prediction
            relation = torch.cat(
                (keys[:, slot], values[:, slot], prediction, error.abs()), dim=-1
            )
            logit = self.write_gate(relation).squeeze(-1)
            retention = self._retention(relation)
            logits.append(logit)
            surprises.append(error.square().mean(dim=-1).sqrt())
            matrix = self._update(
                matrix, keys[:, slot], values[:, slot], logit.sigmoid(), retention
            )
        gate_logits = torch.stack(logits, dim=1).masked_fill(~masks, -torch.inf)
        surprise = torch.stack(surprises, dim=1).masked_fill(~masks, 0.0)
        selected = torch.zeros_like(masks)
        for row in range(len(masks)):
            valid = int(masks[row].sum())
            keep = max(1, round(valid * self.keep_ratio))
            selected[row, gate_logits[row].topk(keep).indices] = True
        probabilities = gate_logits.sigmoid().masked_fill(~masks, 0.0)
        strengths = selected.float() + probabilities - probabilities.detach()
        matrix = torch.zeros_like(matrix)
        final_retentions: list[Tensor] = []
        for slot in range(candidate_features.shape[1]):
            prediction = self._read(matrix, keys[:, slot])
            relation = torch.cat(
                (
                    keys[:, slot],
                    values[:, slot],
                    prediction,
                    (values[:, slot] - prediction).abs(),
                ),
                dim=-1,
            )
            retention = self._retention(relation)
            final_retentions.append(retention)
            matrix = self._update(
                matrix,
                keys[:, slot],
                values[:, slot],
                strengths[:, slot],
                retention,
            )
        return FastWeightState(
            matrix,
            gate_logits,
            selected,
            surprise,
            torch.stack(final_retentions, dim=1),
        )

    def read(self, state: FastWeightState, recall_context: Tensor) -> Tensor:
        """Emit one associated latent using only persistent state and the new query."""
        query = nn.functional.normalize(torch.tanh(self.query_key(recall_context)), dim=-1)
        return nn.functional.normalize(self._read(state.matrix, query), dim=-1)

    def score_candidates(
        self,
        state: FastWeightState,
        candidate_features: Tensor,
        recall_context: Tensor,
    ) -> Tensor:
        """Probe the emitted latent against labeled candidates for evaluation."""
        retrieved = self.read(state, recall_context)
        _, values = self._views(candidate_features)
        cosine = (values * retrieved[:, None, :]).sum(dim=-1)
        logits = cosine * self.logit_scale.exp().clamp(max=100.0)
        return logits.masked_fill(~state.selected, -8.0)

    def forward(
        self,
        candidate_features: Tensor,
        recall_context: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, FastWeightState]:
        state = self.write(candidate_features, masks)
        return self.score_candidates(state, candidate_features, recall_context), state
