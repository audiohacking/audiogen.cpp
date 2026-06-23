import torch
import torch.nn as nn


class LayerNorm(nn.LayerNorm):
    def __init__(self, nout, dim=-1):
        super().__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        if self.dim == -1:
            return super().forward(x)
        return super().forward(x.transpose(1, -1)).transpose(1, -1)


class DurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        n_layers: int = 2,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        padding: str = "SAME"
    ):
        super().__init__()
        self.conv = nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = in_channels if idx == 0 else filter_channels
            self.conv += [
                nn.Sequential(
                    nn.ConstantPad1d(
                        ((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                        if padding == 'SAME' else (kernel_size - 1, 0),
                        0
                    ),
                    nn.Conv1d(
                        in_chans, filter_channels,
                        kernel_size, stride=1, padding=0
                    ),
                    nn.ReLU(),
                    LayerNorm(filter_channels, dim=1),
                    nn.Dropout(p_dropout)
                )
            ]
        self.linear = nn.Linear(filter_channels, 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        x = x.transpose(1, -1)
        x_mask = x_mask.unsqueeze(1).to(x.device)
        for f in self.conv:
            x = f(x)
            x = x * x_mask.float()
        x = self.linear(x.transpose(1, -1)) * x_mask.transpose(1, -1).float()
        return x


class ContentAdapterBase(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.d_out = d_out


class CrossAttentionAdapter(ContentAdapterBase):
    def __init__(
        self,
        d_out: int,
        content_dim: int,
        prefix_dim: int,
        num_heads: int,
        duration_predictor: DurationPredictor,
        dropout: float = 0.1,
        duration_grad_scale: float = 0.1,
    ):
        super().__init__(d_out)
        self.attn = nn.MultiheadAttention(
            embed_dim=content_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=prefix_dim,
            vdim=prefix_dim,
            batch_first=True,
        )
        self.duration_grad_scale = duration_grad_scale
        self.duration_predictor = duration_predictor
        self.global_duration_mlp = nn.Sequential(
            nn.Linear(content_dim, content_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(content_dim, 1)
        )
        self.norm = nn.LayerNorm(content_dim)
        self.content_proj = nn.Conv1d(content_dim, d_out, 1)

    def forward(self, content, content_mask, prefix, prefix_mask):
        attn_output, attn_output_weights = self.attn(
            query=content,
            key=prefix,
            value=prefix,
            key_padding_mask=~prefix_mask.bool()
        )
        attn_output = attn_output * content_mask.unsqueeze(-1).float()
        x = self.norm(attn_output + content)
        x_grad_rescaled = x * self.duration_grad_scale + x.detach() * (
            1 - self.duration_grad_scale
        )
        x_aggregated = (
            x_grad_rescaled * content_mask.unsqueeze(-1).float()
        ).sum(dim=1) / content_mask.sum(dim=1, keepdim=True).float()
        global_duration = self.global_duration_mlp(x_aggregated).squeeze(-1)
        local_duration = self.duration_predictor(
            x_grad_rescaled, content_mask
        ).squeeze(-1)
        content = self.content_proj(x.transpose(1, 2)).transpose(1, 2)
        return content, content_mask, global_duration, local_duration
