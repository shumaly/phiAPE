"""Head-wise RoPE (Su et al., arXiv:2104.09864, equations 14–16).

Both variants share every operation except inverse-frequency selection.
The phiRoPE schedule is the existing phiAPE c=4 rule evaluated at head_dim.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

ROTARY_METHODS = ("RoPE", "phiRoPE")
PHI_ROPE_C = 4.0


def standard_frequencies(length, head_dim, device):
    if head_dim < 2 or head_dim % 2:
        raise ValueError("RoPE requires a positive even per-head dimension")
    return 10000.0 ** (-torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)


def phi_frequencies(length, head_dim, device):
    if length < 2:
        raise ValueError("phiRoPE requires sequence length >= 2, as in the benchmark loaders")
    raw = standard_frequencies(length, head_dim, device)
    normalized = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
    minimum = math.pi / (length - 1)
    maximum = minimum + (head_dim / PHI_ROPE_C) * minimum
    return minimum + normalized * (maximum - minimum)


FREQUENCY_BUILDERS = {"RoPE": standard_frequencies, "phiRoPE": phi_frequencies}


def rotate_pairs(x, frequencies, positions=None):
    """Rotate adjacent pairs of (..., sequence, head_dim) projected Q or K."""
    if x.shape[-1] != 2 * frequencies.numel():
        raise ValueError("Frequency count must equal head_dim // 2")
    if positions is None:
        positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    angles = positions.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * frequencies
    cosine = angles.cos().to(dtype=x.dtype)
    sine = angles.sin().to(dtype=x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


class RotaryMultiheadAttention(nn.MultiheadAttention):
    """Packed projections and initialization match nn.MultiheadAttention.

    This benchmark uses sequence-first self-attention, including bidirectional
    attention and the original attention dropout. V is never rotated. There is
    no added absolute encoding, frequency rescaling, or learned rotary parameter.
    The implementation uses PyTorch 1.13 operations for the original Raven job.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, method="RoPE"):
        super().__init__(embed_dim, num_heads, dropout=dropout)
        if method not in ROTARY_METHODS:
            raise ValueError(f"Unknown rotary method: {method}")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head_dim = d_model / num_heads")
        self.method = method

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if query is not key or query is not value or query.ndim != 3:
            raise ValueError("This benchmark rotary attention supports sequence-first batched self-attention only")
        length, batch, width = query.shape
        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        def split_heads(tensor):
            return tensor.reshape(length, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        q, k, v = (split_heads(tensor) for tensor in (q, k, v))
        frequencies = FREQUENCY_BUILDERS[self.method](length, self.head_dim, query.device)
        q, k = rotate_pairs(q, frequencies), rotate_pairs(k, frequencies)
        scores = torch.matmul(q * (self.head_dim ** -0.5), k.transpose(-2, -1))
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.ndim == 3:
                mask = attn_mask.reshape(batch, self.num_heads, length, length)
            else:
                raise ValueError("attn_mask must have shape (L,L) or (batch*heads,L,L)")
            scores = scores.masked_fill(mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
        if is_causal:
            mask = torch.ones(length, length, dtype=torch.bool, device=query.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        weights = F.dropout(torch.softmax(scores, dim=-1), p=self.dropout, training=self.training)
        context = torch.matmul(weights, v).permute(2, 0, 1, 3).contiguous().reshape(length, batch, width)
        output = F.linear(context, self.out_proj.weight, self.out_proj.bias)
        if not need_weights:
            return output, None
        return output, weights.mean(dim=1) if average_attn_weights else weights
