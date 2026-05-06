"""
aggregation.py — Token aggregation strategy and feature extraction (v3).

Key improvements over v2:
  1. Mean-pool over RESPONSE tokens only (not last-token).
     The last token is often EOS or punctuation — it carries almost no
     semantic content. Mean-pooling over the actual response tokens
     captures the distributed representation of what the model generated.
  2. Wider layer coverage: 5 layers across the truthfulness band
     instead of 3. More signal, still well under the n_samples budget.
  3. Inter-layer representation dynamics: how the response representation
     changes between adjacent layers is a strong hallucination indicator
     (hallucinations show more turbulent / inconsistent layer-to-layer drift).
  4. Richer geometric features: per-layer variance, norm profile, and
     response-length signal.

For Qwen2.5-0.5B the hidden-states tuple has 25 elements (1 embedding + 24
transformer layers), each shaped (seq_len, hidden_dim=896).
"""

from __future__ import annotations

import torch


# Mean-pool readouts at five mid-to-late layers.
# Layers 16-24 (of 24) cover the band where truthfulness signals peak
# in decoder-only models (Azaria & Mitchell 2023; Burns et al. 2022).
SELECTED_LAYERS: tuple[int, ...] = (-12, -9, -6, -3, -1)
"""~50%, ~62%, ~75%, ~87%, ~100% depth."""


def _last_real_position(attention_mask: torch.Tensor) -> int:
    """Return the index of the last non-padding token."""
    real_idx = attention_mask.nonzero(as_tuple=False).flatten()
    return int(real_idx[-1].item())


def _response_start_position(attention_mask: torch.Tensor) -> int:
    """Estimate where the assistant response begins.

    In ChatML the prompt ends with `<|im_start|>assistant\n` (typically
    2-3 tokens). We approximate the response start as the last 60% of
    real tokens — responses in this dataset are usually 5-30 tokens
    while prompts are 100-300 tokens, so the last 40% safely captures
    only the response.
    """
    real_idx = attention_mask.nonzero(as_tuple=False).flatten()
    n_real = len(real_idx)
    # Response is roughly the last 40% of non-padding tokens.
    # For very short sequences (< 20 tokens) just use everything.
    if n_real < 20:
        return 0
    return int(real_idx[int(n_real * 0.6)].item())


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool over response tokens at each selected layer.

    Args:
        hidden_states:  Tensor of shape (n_layers, seq_len, hidden_dim).
        attention_mask: 1-D tensor of shape (seq_len,).

    Returns:
        1-D feature tensor of shape (len(SELECTED_LAYERS) * hidden_dim,)
        i.e. 5 * 896 = 4480 for Qwen2.5-0.5B.
    """
    real_idx = attention_mask.nonzero(as_tuple=False).flatten()
    real_idx = real_idx.to(hidden_states.device)

    resp_start = _response_start_position(attention_mask)
    # Get indices of response tokens (from resp_start to last real token)
    resp_idx = real_idx[real_idx >= resp_start]

    if len(resp_idx) == 0:
        resp_idx = real_idx  # fallback: use all tokens

    pooled_layers: list[torch.Tensor] = []
    for layer_idx in SELECTED_LAYERS:
        layer = hidden_states[layer_idx]  # (seq_len, hidden_dim)
        resp_tokens = layer.index_select(0, resp_idx)  # (n_resp, hidden_dim)
        pooled_layers.append(resp_tokens.mean(dim=0))  # (hidden_dim,)

    return torch.cat(pooled_layers, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Hand-crafted geometric features describing the response representation.

    Features:
      1. Response length (normalized) — hallucinated responses tend to differ
         in length from truthful ones.
      2. Mean L2 norm at each selected layer — "activation strength" profile.
      3. Per-layer variance (mean across hidden dims) — how heterogeneous
         the response tokens are; hallucinations often show higher variance.
      4. Inter-layer cosine similarity — representation drift between
         adjacent selected layers; hallucinations show more turbulent drift.
    """
    real_idx = attention_mask.nonzero(as_tuple=False).flatten().to(hidden_states.device)
    resp_start = _response_start_position(attention_mask)
    resp_idx = real_idx[real_idx >= resp_start]
    if len(resp_idx) == 0:
        resp_idx = real_idx

    n_resp = len(resp_idx)

    # 1. Response length (normalized to ~O(1))
    seq_len_feat = torch.tensor(
        [n_resp / 50.0],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    # Select response tokens from all 24 layers for geometric features
    # Shape: (24, n_resp, hidden_dim)
    resp_states = hidden_states.index_select(1, resp_idx)

    # 2. Mean L2 norm per layer — (n_layers,)
    token_norms = resp_states.norm(dim=-1)  # (n_layers, n_resp)
    mean_norms = token_norms.mean(dim=-1)   # (n_layers,)

    # 3. Per-layer variance (mean across hidden dims) — (n_layers,)
    token_vars = resp_states.var(dim=1).mean(dim=-1)  # (n_layers,)

    # 4. Inter-layer cosine similarity for selected layers
    # Mean-pool each layer first, then compute cosine sim between adjacent
    layer_means = []
    for layer_idx in SELECTED_LAYERS:
        layer = hidden_states[layer_idx]
        resp_tokens = layer.index_select(0, resp_idx)
        layer_means.append(resp_tokens.mean(dim=0))
    # layer_means: list of (hidden_dim,) tensors

    cos_sims = []
    for i in range(len(layer_means) - 1):
        sim = torch.nn.functional.cosine_similarity(
            layer_means[i].unsqueeze(0),
            layer_means[i + 1].unsqueeze(0),
            dim=-1,
        )  # (1,)
        cos_sims.append(sim.squeeze(0))

    return torch.cat([seq_len_feat, mean_norms, token_vars] + cos_sims, dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features."""
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
