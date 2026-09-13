from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def pool_last_token(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the final non-padding token from a causal language model."""
    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must have shape [batch, tokens, features]")
    if attention_mask.ndim != 2 or attention_mask.shape != last_hidden_state.shape[:2]:
        raise ValueError("attention_mask must match the first two hidden-state dimensions")
    active = attention_mask.bool()
    if bool((~active.any(dim=-1)).any()):
        raise ValueError("every sequence must contain at least one unmasked token")
    rows = torch.arange(len(last_hidden_state), device=last_hidden_state.device)
    positions = torch.arange(last_hidden_state.shape[1], device=last_hidden_state.device)
    final_positions = positions.expand_as(active).masked_fill(~active, -1).max(dim=-1).values
    return last_hidden_state[rows, final_positions]


def pool_mean(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool non-padding token states."""
    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must have shape [batch, tokens, features]")
    if attention_mask.ndim != 2 or attention_mask.shape != last_hidden_state.shape[:2]:
        raise ValueError("attention_mask must match the first two hidden-state dimensions")
    weights = attention_mask.to(dtype=last_hidden_state.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1)
    if bool((counts < 1).any()):
        raise ValueError("every sequence must contain at least one unmasked token")
    return (last_hidden_state * weights).sum(dim=1) / counts


def encode_with_gemma(
    texts: Sequence[str],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
    pooling: str = "last",
) -> Tensor:
    """Encode text with a frozen Gemma backbone and normalized final-token states."""
    if batch_size < 1 or max_length < 1:
        raise ValueError("batch_size and max_length must be positive")
    if pooling not in {"last", "mean"}:
        raise ValueError(f"unknown Gemma pooling: {pooling}")

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Gemma 3 270M produces non-finite hidden states in float16 on MPS.
    # Keep the frozen feature extractor in float32; the resulting artifact is
    # generated once and the downstream memory layer remains inexpensive.
    model = AutoModel.from_pretrained(model_name, dtype=torch.float32).to(device)
    model.eval()
    chunks: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            pool = pool_last_token if pooling == "last" else pool_mean
            pooled = pool(hidden, encoded["attention_mask"])
            pooled = F.normalize(pooled.float(), dim=-1)
            if not bool(torch.isfinite(pooled).all()):
                raise RuntimeError("Gemma produced non-finite pooled features")
            chunks.append(pooled.cpu())
    return torch.cat(chunks)


def encode_gemma_layers(
    texts: Sequence[str],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
    layers: Sequence[int],
    pooling: str = "mean",
) -> dict[int, Tensor]:
    """Encode several frozen Gemma hidden layers in one model pass."""
    if batch_size < 1 or max_length < 1:
        raise ValueError("batch_size and max_length must be positive")
    if pooling not in {"last", "mean"}:
        raise ValueError(f"unknown Gemma pooling: {pooling}")
    requested = tuple(dict.fromkeys(layers))
    if not requested:
        raise ValueError("at least one Gemma layer is required")

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=torch.float32).to(device)
    model.eval()
    hidden_count = model.config.num_hidden_layers + 1
    resolved = {
        layer: layer if layer >= 0 else hidden_count + layer for layer in requested
    }
    if any(index < 0 or index >= hidden_count for index in resolved.values()):
        raise ValueError(
            f"Gemma layers must resolve inside [0, {hidden_count - 1}]: {requested}"
        )

    chunks: dict[int, list[Tensor]] = {layer: [] for layer in requested}
    pool = pool_last_token if pooling == "last" else pool_mean
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            states = model(**encoded, output_hidden_states=True).hidden_states
            for layer, index in resolved.items():
                pooled = F.normalize(
                    pool(states[index], encoded["attention_mask"]).float(), dim=-1
                )
                if not bool(torch.isfinite(pooled).all()):
                    raise RuntimeError(f"Gemma layer {layer} produced non-finite features")
                chunks[layer].append(pooled.cpu())
    return {layer: torch.cat(parts) for layer, parts in chunks.items()}


def build_echo_inputs(
    tokenizer: object,
    texts: Sequence[str],
    *,
    max_length: int,
    device: str,
) -> tuple[dict[str, Tensor], Tensor]:
    """Repeat each sequence and mark only its second occurrence for pooling."""
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise ValueError("Gemma tokenizer must define a padding token")
    separator = tokenizer.encode("\n\n", add_special_tokens=False)  # type: ignore[attr-defined]
    prefix = [] if bos_token_id is None else [bos_token_id]
    copy_length = (max_length - len(prefix) - len(separator)) // 2
    if copy_length < 1:
        raise ValueError("max_length is too small for echo encoding")
    tokenized = tokenizer(  # type: ignore[operator]
        list(texts),
        add_special_tokens=False,
        truncation=True,
        max_length=copy_length,
    )["input_ids"]
    sequences: list[list[int]] = []
    echo_masks: list[list[int]] = []
    for tokens in tokenized:
        sequence = [*prefix, *tokens, *separator, *tokens]
        sequences.append(sequence)
        echo_masks.append([0] * (len(prefix) + len(tokens) + len(separator)) + [1] * len(tokens))
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    echo_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, (sequence, selected) in enumerate(zip(sequences, echo_masks, strict=True)):
        input_ids[row, : len(sequence)] = torch.tensor(sequence)
        attention_mask[row, : len(sequence)] = 1
        echo_mask[row, : len(sequence)] = torch.tensor(selected)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }, echo_mask.to(device)


def encode_with_gemma_echo(
    texts: Sequence[str],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
    layer: int = -1,
) -> Tensor:
    """Create echo embeddings from the second copy seen by a causal Gemma model."""
    if batch_size < 1 or max_length < 1:
        raise ValueError("batch_size and max_length must be positive")

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=torch.float32).to(device)
    model.eval()
    hidden_count = model.config.num_hidden_layers + 1
    resolved_layer = layer if layer >= 0 else hidden_count + layer
    if resolved_layer < 0 or resolved_layer >= hidden_count:
        raise ValueError(f"Gemma layer must resolve inside [0, {hidden_count - 1}]: {layer}")
    chunks: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded, echo_mask = build_echo_inputs(
                tokenizer,
                texts[start : start + batch_size],
                max_length=max_length,
                device=device,
            )
            if resolved_layer == hidden_count - 1:
                hidden = model(**encoded).last_hidden_state
            else:
                hidden = model(**encoded, output_hidden_states=True).hidden_states[resolved_layer]
            pooled = F.normalize(pool_mean(hidden, echo_mask).float(), dim=-1)
            if not bool(torch.isfinite(pooled).all()):
                raise RuntimeError("Gemma echo encoder produced non-finite features")
            chunks.append(pooled.cpu())
    return torch.cat(chunks)
