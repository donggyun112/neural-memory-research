from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


RecallMode = Literal["candidate", "query", "joint"]
SimilarityMode = Literal["none", "feature", "residual"]


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
        similarity: SimilarityMode = "none",
        center_similarity: bool = True,
        correction_bound: float = 0.0,
        write_context: bool = False,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or memory_dim < 1:
            raise ValueError("dimensions must be positive")
        if not 0.0 < keep_ratio <= provisional_ratio <= 1.0:
            raise ValueError("ratios must satisfy 0 < keep_ratio <= provisional_ratio <= 1")
        if similarity not in ("none", "feature", "residual"):
            raise ValueError(f"unknown similarity mode: {similarity}")
        self.provisional_ratio = provisional_ratio
        self.keep_ratio = keep_ratio
        # The write stage trains candidate_projection for write-worthiness and
        # then freezes it, so a trace may no longer carry what stage two needs to
        # match the later event. Both non-default modes hand stage two the
        # untouched frozen-encoder similarity instead of making it rediscover
        # one; they differ in how hard that signal is to reach. As one input of
        # 4*memory_dim+1 a scalar starts out negligible, which "residual" avoids
        # by scoring with it directly and letting the head learn a correction.
        self.similarity = similarity
        self.center_similarity = center_similarity
        # Write training reshapes candidate_projection around write-worthiness,
        # and a consolidation head reading those traces learns a correction large
        # enough to bury the similarity term: alone each stage is harmless, but
        # together they cost 0.325 retention. A bounded correction keeps the
        # similarity in charge and lets the head only adjust.
        self.correction_bound = correction_bound
        self.candidate_projection = nn.Linear(feature_dim, memory_dim)
        self.event_projection = nn.Linear(feature_dim, memory_dim)
        self.recall_projection = nn.Linear(feature_dim, memory_dim)
        # Choosing k of n is comparative, but a gate scoring each candidate in
        # isolation cannot express "more promising than the rest of this
        # episode". The context form pairs every trace with the episode mean.
        self.write_context = write_context
        self.write_gate = nn.Sequential(
            nn.Linear(memory_dim * 4 if write_context else memory_dim, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, 1),
        )
        self.consolidation_head = nn.Sequential(
            nn.Linear(memory_dim * 4 + int(similarity == "feature"), memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, 1),
        )
        self.recall_head = nn.Sequential(
            nn.Linear(memory_dim * 4, memory_dim), nn.GELU(), nn.Linear(memory_dim, 1)
        )
        # Cosine gaps between a matching and a non-matching candidate are about
        # 0.25, so the scale converts them into a usable logit margin. Starting
        # the head at zero makes the untrained model exactly the cosine rule,
        # which is the baseline it previously failed to reach.
        self.similarity_scale = nn.Parameter(torch.tensor(8.0))
        if similarity == "residual":
            nn.init.zeros_(self.consolidation_head[-1].weight)
            nn.init.zeros_(self.consolidation_head[-1].bias)

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
        if self.write_context:
            weights = masks.to(dtype=traces.dtype)[..., None]
            episode = (traces * weights).sum(dim=1, keepdim=True) / weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            scored = self._relation(traces, episode.expand_as(traces))
        else:
            scored = traces
        write_logits = self.write_gate(scored).squeeze(-1)
        provisional, strengths = hard_top_k(write_logits, masks, self.provisional_ratio)
        return DeferredTraceState(
            traces=traces * strengths[..., None],
            write_logits=write_logits,
            provisional=provisional,
            consolidation_logits=torch.zeros_like(write_logits),
            selected=provisional,
            strengths=strengths,
        )

    def consolidate(
        self, state: DeferredTraceState, event: Tensor, similarity: Tensor | None = None
    ) -> DeferredTraceState:
        """Narrow the provisional set using one later event, keeping the discard final."""
        if event.ndim != 2:
            raise ValueError("event must have shape [batch, feature_dim]")
        relation = self._paired(state.traces, event, self.event_projection)
        if self.similarity != "none" and similarity is None:
            raise ValueError("this model was built to consume a similarity feature")
        if self.similarity == "feature":
            relation = torch.cat((relation, similarity[..., None]), dim=-1)
        logits = self.consolidation_head(relation).squeeze(-1)
        if self.correction_bound > 0.0:
            logits = self.correction_bound * torch.tanh(logits)
        if self.similarity == "residual":
            centred = similarity
            if self.center_similarity:
                # Encoder cosines are all positive, so an uncentred residual makes
                # every candidate score high and forces the head to learn a large
                # negative offset, which is where the ranking gets destroyed.
                # Subtracting the per-episode mean is rank-preserving, so the
                # untrained model is unchanged.
                centred = similarity - similarity.mean(dim=-1, keepdim=True)
            logits = logits + self.similarity_scale * centred
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

    @staticmethod
    def encoder_similarity(candidate_features: Tensor, event: Tensor) -> Tensor:
        """Frozen-encoder similarity between each candidate and the later event."""
        return torch.einsum("etf,ef->et", candidate_features, event)

    def forward(
        self,
        candidate_features: Tensor,
        event: Tensor,
        query: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, DeferredTraceState]:
        similarity = (
            self.encoder_similarity(candidate_features, event)
            if self.similarity != "none"
            else None
        )
        state = self.consolidate(self.write(candidate_features, masks), event, similarity)
        return self.recall(state, query), state


@dataclass(frozen=True)
class AssociativeState:
    matrix: Tensor
    eligibility: Tensor
    values: Tensor
    write_strengths: Tensor
    event_gains: Tensor


class AssociativeDeferredMemory(nn.Module):
    """Memory as a changing weight matrix, with no slot selection anywhere.

    Every candidate is written with a graded strength; nothing is discarded and
    there is no top-k. Capacity is the matrix itself, so what survives is decided
    by interference and decay rather than by an admission decision. A later event
    then modulates the still-eligible traces the way a reinforcement pathway does:
    it never sees the stored content, only how strongly each recent trace responds
    to it.

    Persistent state is ``memory_dim ** 2`` scalars. Eligibility is transient and
    is dropped once the event has acted, so it is not part of the stored memory.
    """

    def __init__(
        self,
        feature_dim: int,
        memory_dim: int = 12,
        decay: float = 0.99,
        event_gain_bound: float = 1.0,
        competitive_gain: bool = True,
        tie_keys: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or memory_dim < 1:
            raise ValueError("dimensions must be positive")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be inside (0, 1]")
        self.memory_dim = memory_dim
        self.event_gain_bound = event_gain_bound
        self.competitive_gain = competitive_gain
        self.tie_keys = tie_keys
        self.candidate_key = nn.Linear(feature_dim, memory_dim)
        self.candidate_value = nn.Linear(feature_dim, memory_dim)
        self.query_key = nn.Linear(feature_dim, memory_dim)
        self.event_key = nn.Linear(feature_dim, memory_dim)
        self.write_strength = nn.Sequential(
            nn.Linear(memory_dim * 3, memory_dim), nn.GELU(), nn.Linear(memory_dim, 1)
        )
        self.decay_logit = nn.Parameter(torch.logit(torch.tensor(decay)))
        # The modulatory gain reads only how strongly a trace answers the event,
        # never the trace's content, which is what makes it a separate pathway.
        self.event_gain = nn.Sequential(
            nn.Linear(3, memory_dim), nn.GELU(), nn.Linear(memory_dim, 1)
        )
        nn.init.zeros_(self.event_gain[-1].weight)
        nn.init.zeros_(self.event_gain[-1].bias)

    def _views(self, features: Tensor) -> tuple[Tensor, Tensor]:
        keys = F.normalize(torch.tanh(self.candidate_key(features)), dim=-1)
        values = F.normalize(torch.tanh(self.candidate_value(features)), dim=-1)
        return keys, values

    @staticmethod
    def _read(matrix: Tensor, key: Tensor) -> Tensor:
        return torch.bmm(matrix, key[..., None]).squeeze(-1)

    def write(self, candidate_features: Tensor, masks: Tensor) -> AssociativeState:
        if candidate_features.ndim != 3:
            raise ValueError("candidates must have shape [batch, traces, feature_dim]")
        if masks.shape != candidate_features.shape[:2]:
            raise ValueError("masks must match candidate trace axes")
        keys, values = self._views(candidate_features)
        batch, traces, _ = candidate_features.shape
        matrix = candidate_features.new_zeros(batch, self.memory_dim, self.memory_dim)
        eligibility = candidate_features.new_zeros(
            batch, traces, self.memory_dim, self.memory_dim
        )
        decay = self.decay_logit.sigmoid()
        strengths: list[Tensor] = []
        for slot in range(traces):
            key, value = keys[:, slot], values[:, slot]
            prediction = self._read(matrix, key)
            strength = self.write_strength(
                torch.cat((key, value, value - prediction), dim=-1)
            ).squeeze(-1).sigmoid()
            strength = strength * masks[:, slot].to(strength.dtype)
            delta = (value - prediction)[..., :, None] * key[..., None, :]
            eligibility[:, slot] = strength[..., None, None] * delta
            matrix = decay * matrix + eligibility[:, slot]
            strengths.append(strength)
        written = torch.stack(strengths, dim=1)
        return AssociativeState(
            matrix=matrix,
            eligibility=eligibility,
            values=values,
            write_strengths=written,
            event_gains=torch.zeros_like(written),
        )

    def consolidate(self, state: AssociativeState, event: Tensor) -> AssociativeState:
        """Let a later event strengthen or depress the traces that answer to it."""
        if event.ndim != 2:
            raise ValueError("event must have shape [batch, feature_dim]")
        key = F.normalize(torch.tanh(self.event_key(event)), dim=-1)
        response = torch.einsum("btij,bj->bti", state.eligibility, key)
        # Coincidence, not content: a trace is addressed when the event's key
        # retrieves something aligned with what that trace stored. The signed
        # alignment is what makes the pathway selective; a magnitude alone cannot
        # tell an answering trace from a merely large one.
        alignment = (F.normalize(response, dim=-1) * state.values).sum(dim=-1)
        summary = torch.stack(
            (alignment, response.norm(dim=-1), state.write_strengths), dim=-1
        )
        gains = self.event_gain_bound * torch.tanh(self.event_gain(summary).squeeze(-1))
        if self.competitive_gain:
            # A uniform gain only rescales the matrix, which cannot reorder what
            # it retrieves, so an unconstrained pathway degenerates to a global
            # scale and does nothing. Forcing the gains to sum to zero makes
            # consolidation competitive: strengthening one trace has to come out
            # of another, which is the allocation rule phase 1 was built on.
            gains = gains - gains.mean(dim=-1, keepdim=True)
        matrix = state.matrix + torch.einsum("bt,btij->bij", gains, state.eligibility)
        return AssociativeState(
            matrix=matrix,
            eligibility=state.eligibility,
            values=state.values,
            write_strengths=state.write_strengths,
            event_gains=gains,
        )

    def _probe(self, features: Tensor) -> Tensor:
        """Address the matrix. Tying this to the candidate key makes the query and
        its target share a space by construction instead of having to discover one:
        the frozen encoder already puts them at cosine 0.61 against 0.43 for the
        rest, and a separate projection throws that away."""
        projection = self.candidate_key if self.tie_keys else self.query_key
        return F.normalize(torch.tanh(projection(features)), dim=-1)

    def recall(self, state: AssociativeState, query: Tensor, candidates: Tensor) -> Tensor:
        """Score candidates by how well the matrix answers the query with them."""
        retrieved = F.normalize(self._read(state.matrix, self._probe(query)), dim=-1)
        _, values = self._views(candidates)
        return torch.einsum("btd,bd->bt", F.normalize(values, dim=-1), retrieved)

    def forward(
        self, candidate_features: Tensor, event: Tensor, query: Tensor, masks: Tensor
    ) -> tuple[Tensor, AssociativeState]:
        state = self.consolidate(self.write(candidate_features, masks), event)
        return self.recall(state, query, candidate_features), state


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
