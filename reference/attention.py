import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange, repeat
from inspect import isfunction

from .modules import RMSNorm


# --- Rotary Position Embeddings ---

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=-2):
        seq_len = x.shape[seq_dimension]
        if (
            seq_len != self._seq_len_cached
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self._cos_cached = emb.cos()[None, None, :, :].to(x.dtype)
            self._sin_cached = emb.sin()[None, None, :, :].to(x.dtype)
        return self._cos_cached, self._sin_cached

    def forward(self, q, k):
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            q.float(), seq_dimension=-2
        )
        if k is not None:
            return (
                apply_rotary_pos_emb(q.float(), self._cos_cached, self._sin_cached).type_as(q),
                apply_rotary_pos_emb(k.float(), self._cos_cached, self._sin_cached).type_as(k),
            )
        else:
            return (
                apply_rotary_pos_emb(q.float(), self._cos_cached, self._sin_cached).type_as(q),
                None,
            )


# --- Attention Helpers ---

def add_mask(sim, mask):
    b, ndim = sim.shape[0], mask.ndim
    if ndim == 3:
        mask = rearrange(mask, "b n m -> b 1 n m")
    if ndim == 2:
        mask = repeat(mask, "n m -> b 1 n m", b=b)
    max_neg_value = -torch.finfo(sim.dtype).max
    sim = sim.masked_fill(~mask, max_neg_value)
    return sim


def create_mask(q_shape, k_shape, device, q_mask=None, k_mask=None):
    def default(val, d):
        return val if val is not None else (d() if isfunction(d) else d)

    b, i, j = q_shape[0], q_shape[-2], k_shape[-2]
    q_mask = default(q_mask, torch.ones((b, i), device=device, dtype=torch.bool))
    k_mask = default(k_mask, torch.ones((b, j), device=device, dtype=torch.bool))
    attn_mask = rearrange(q_mask, "b i -> b 1 i 1") * rearrange(k_mask, "b j -> b 1 1 j")
    return attn_mask


# --- Main Attention Module ---

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        context_dim=None,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        qk_norm=None,
        attn_drop=0.0,
        proj_drop=0.0,
        rope_mode="none",
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.cross_attn = context_dim is not None
        context_dim = dim if context_dim is None else context_dim

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, dim, bias=qkv_bias)

        if qk_norm is None:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        elif qk_norm == "layernorm":
            self.norm_q = nn.LayerNorm(head_dim)
            self.norm_k = nn.LayerNorm(head_dim)
        elif qk_norm == "rmsnorm":
            self.norm_q = RMSNorm(head_dim)
            self.norm_k = RMSNorm(head_dim)
        else:
            raise NotImplementedError

        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.cross_attn:
            assert rope_mode == "none"
        self.rope_mode = rope_mode
        if self.rope_mode == "shared" or self.rope_mode == "x_only":
            self.rotary = RotaryEmbedding(dim=head_dim)

    def _rotary(self, q, k, extras):
        if self.rope_mode == "shared":
            q, k = self.rotary(q=q, k=k)
        elif self.rope_mode == "x_only":
            q_x, k_x = self.rotary(q=q[:, :, extras:, :], k=k[:, :, extras:, :])
            q_c, k_c = q[:, :, :extras, :], k[:, :, :extras, :]
            q = torch.cat((q_c, q_x), dim=2)
            k = torch.cat((k_c, k_x), dim=2)
        elif self.rope_mode == "none":
            pass
        else:
            raise NotImplementedError
        return q, k

    def _attn(self, q, k, v, mask_binary):
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0,
            attn_mask=mask_binary,
        )
        x = einops.rearrange(x, "B H L D -> B L (H D)")
        return x

    def forward(self, x, context=None, context_mask=None, extras=0):
        B, L, C = x.shape
        if context is None:
            context = x

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        if context_mask is not None:
            mask_binary = create_mask(x.shape, context.shape, x.device, None, context_mask)
        else:
            mask_binary = None

        q = einops.rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
        k = einops.rearrange(k, "B L (H D) -> B H L D", H=self.num_heads)
        v = einops.rearrange(v, "B L (H D) -> B H L D", H=self.num_heads)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q, k = self._rotary(q, k, extras)

        x = self._attn(q, k, v, mask_binary)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
