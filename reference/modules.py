import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange


def trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def film_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


def unpatchify(x, channels=3, input_type="2d", img_size=None):
    if input_type == "2d":
        patch_size = int((x.shape[2] // channels) ** 0.5)
        h, w = img_size[0] // patch_size, img_size[1] // patch_size
        x = rearrange(
            x,
            "B (h w) (p1 p2 C) -> B C (h p1) (w p2)",
            h=h,
            p1=patch_size,
            p2=patch_size,
        )
    elif input_type == "1d":
        patch_size = int(x.shape[2] // channels)
        h = x.shape[1]
        x = rearrange(x, "B h (p1 C) -> B C (h p1)", h=h, p1=patch_size)
    return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, out_size=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size).type(
            self.mlp[0].weight.dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans=3, embed_dim=768, input_type="2d"):
        super().__init__()
        self.patch_size = patch_size
        self.input_type = input_type
        if input_type == "2d":
            self.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True
            )
        elif input_type == "1d":
            self.proj = nn.Conv1d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True
            )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PE_wrapper(nn.Module):
    def __init__(self, dim=768, method="abs", length=None, **kwargs):
        super().__init__()
        self.method = method
        if method == "abs":
            self.length = length
            self.abs_pe = nn.Parameter(torch.zeros(1, length, dim))
            trunc_normal_(self.abs_pe, mean=0.0, std=0.02, a=-0.04, b=0.04)
        elif method == "none":
            self.id = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.method == "abs":
            _, L, _ = x.shape
            assert L <= self.length
            x = x + self.abs_pe[:, :L, :]
        elif self.method == "none":
            x = self.id(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate, approximate=self.approximate)
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(
            dtype=gate.dtype
        )

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate)
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        dropout=0.0,
        activation_fn="geglu",
        final_dropout=False,
        inner_dim=None,
        bias=True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        else:
            raise NotImplementedError

        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
