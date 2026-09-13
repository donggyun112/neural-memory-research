from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def build_memory_prompt(
    sessions: Sequence[str], question: str, *, max_session_chars: int = 2000
) -> str:
    """Render selected memory traces and the question as one generator prompt."""
    if max_session_chars < 1:
        raise ValueError("max_session_chars must be positive")
    kept = [session.strip()[:max_session_chars] for session in sessions if session.strip()]
    blocks = [f"Memory {index + 1}:\n{session}" for index, session in enumerate(kept)]
    if not blocks:
        return f"Question: {question}\nAnswer:"
    return "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


def scored_token_targets(
    input_ids: Tensor, answer_mask: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Locate every answer token and the state position that must predict it."""
    if input_ids.shape != answer_mask.shape or input_ids.ndim != 2:
        raise ValueError("input_ids and answer_mask must share shape [batch, tokens]")
    scored = answer_mask[:, 1:].bool()
    if bool((~scored.any(dim=1)).any()):
        raise ValueError("every sequence must contain at least one scored answer token")
    rows, positions = scored.nonzero(as_tuple=True)
    return rows, positions, input_ids[:, 1:][rows, positions]


def mean_nll_per_row(token_losses: Tensor, rows: Tensor, batch: int) -> Tensor:
    """Average per-token losses back into one score per sequence."""
    totals = torch.zeros(batch, dtype=token_losses.dtype, device=token_losses.device)
    counts = torch.zeros_like(totals)
    totals.index_add_(0, rows, token_losses)
    counts.index_add_(0, rows, torch.ones_like(token_losses))
    if bool((counts < 1).any()):
        raise ValueError("every sequence must contain at least one scored answer token")
    return totals / counts


def answer_token_nll(logits: Tensor, input_ids: Tensor, answer_mask: Tensor) -> Tensor:
    """Mean negative log-likelihood per answer token under a causal language model."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, tokens, vocabulary]")
    if input_ids.shape != logits.shape[:2]:
        raise ValueError("input_ids must match the logits batch and length")
    rows, positions, targets = scored_token_targets(input_ids, answer_mask)
    selected = logits[:, :-1][rows, positions].float()
    losses = F.cross_entropy(selected, targets, reduction="none")
    return mean_nll_per_row(losses, rows, len(input_ids))


def _encode_pair(
    tokenizer: object, prompt: str, answer: str, *, max_length: int
) -> tuple[list[int], list[int], bool]:
    """Tokenize prompt and answer, trimming the prompt head so the answer survives.

    The third value reports whether the prompt lost tokens, because a silently
    truncated prompt can delete the evidence a condition is meant to supply.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]  # type: ignore[operator]
    answer_ids = tokenizer(f" {answer.strip()}", add_special_tokens=False)["input_ids"]  # type: ignore[operator]
    if not answer_ids:
        raise ValueError("answer must tokenize to at least one token")
    answer_ids = answer_ids[: max(1, max_length // 2)]
    budget = max_length - len(answer_ids)
    if budget < 1:
        raise ValueError("max_length is too small for the answer")
    # Keep the tail of the prompt: the question sits next to the answer.
    truncated = len(prompt_ids) > budget
    return prompt_ids[-budget:], answer_ids, truncated


def score_answer_nll(
    prompts: Sequence[str],
    answers: Sequence[str],
    *,
    model_name: str,
    device: str,
    batch_size: int = 4,
    max_length: int = 2048,
    dtype: str = "float32",
) -> tuple[Tensor, int]:
    """Score each gold answer with a frozen causal language model.

    Returns the per-episode answer likelihood and how many prompts were cut to
    fit ``max_length``.
    """
    if len(prompts) != len(answers):
        raise ValueError("prompts and answers must have equal length")
    if batch_size < 1 or max_length < 2:
        raise ValueError("batch_size and max_length must be positive")
    if dtype not in {"float32", "bfloat16"}:
        raise ValueError(f"unsupported generator dtype: {dtype}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("generator tokenizer must define a padding token")
    # Gemma 3 270M produces non-finite states in float16 on MPS, so float32 is
    # the default; bfloat16 keeps a multi-billion-parameter generator in memory.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=getattr(torch, dtype)
    ).to(device)
    model.eval()
    softcap = getattr(model.config, "final_logit_softcapping", None)

    scores: list[Tensor] = []
    truncated_prompts = 0
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            pairs = [
                _encode_pair(tokenizer, prompt, answer, max_length=max_length)
                for prompt, answer in zip(
                    prompts[start : start + batch_size],
                    answers[start : start + batch_size],
                    strict=True,
                )
            ]
            truncated_prompts += sum(1 for *_, truncated in pairs if truncated)
            width = max(len(prompt) + len(answer) for prompt, answer, _ in pairs)
            input_ids = torch.full((len(pairs), width), pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(pairs), width), dtype=torch.long)
            answer_mask = torch.zeros((len(pairs), width), dtype=torch.long)
            for row, (prompt_ids, answer_ids, _) in enumerate(pairs):
                sequence = [*prompt_ids, *answer_ids]
                input_ids[row, : len(sequence)] = torch.tensor(sequence)
                attention_mask[row, : len(sequence)] = 1
                answer_mask[row, len(prompt_ids) : len(sequence)] = 1
            rows, positions, targets = scored_token_targets(input_ids, answer_mask)
            hidden = model.model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            ).last_hidden_state
            # Gemma's vocabulary is too large to materialize logits for every
            # position on MPS; project only the states that predict an answer token.
            selected = model.lm_head(hidden[rows.to(device), positions.to(device)]).float()
            if softcap:
                selected = torch.tanh(selected / softcap) * softcap
            losses = F.cross_entropy(selected, targets.to(device), reduction="none")
            batch_scores = mean_nll_per_row(losses, rows.to(device), len(pairs))
            if not bool(torch.isfinite(batch_scores).all()):
                raise RuntimeError("generator produced non-finite answer likelihoods")
            scores.append(batch_scores.float().cpu())
    return torch.cat(scores), truncated_prompts
