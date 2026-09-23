import torch
import torch.nn as nn
import math
from typing import Tuple


class MultiScaleConvBlock(nn.Module):
    """
    Multi-scale residual convolution block for ECG morphology:
    - small kernel: QRS/high-frequency morphology
    - mid kernel: P/T morphology
    - large kernel: slow morphology changes
    - dilated kernel: broader context without pooling
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.branch_small = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.branch_mid = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.branch_large = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=31, padding=15, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.branch_dilated = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=15,
                padding=14,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.merge = nn.Sequential(
            nn.Conv1d(out_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.res_proj = nn.Identity() if in_channels == out_channels else nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_proj(x)
        x1 = self.branch_small(x)
        x2 = self.branch_mid(x)
        x3 = self.branch_large(x)
        x4 = self.branch_dilated(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.merge(x)
        return self.act(x + residual)


class MorphologyEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = []
        c_in = in_channels
        for _ in range(num_layers):
            layers.append(MultiScaleConvBlock(c_in, hidden_dim))
            c_in = hidden_dim
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TemporalTokenizer(nn.Module):
    def __init__(self, in_dim: int, token_dim: int, stride: int):
        super().__init__()
        self.proj = nn.Conv1d(
            in_dim,
            token_dim,
            kernel_size=stride,
            stride=stride,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.transpose(1, 2)


def get_sinusoidal_pos_embed(n_seq: int, d_hid: int):
    embeddings = torch.zeros(n_seq, d_hid)
    position = torch.arange(0, n_seq, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    return embeddings.unsqueeze(0)


class RhythmTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, depth: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(w * x, dim=1)


class ECGHybridEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        cnn_hidden_dim: int = 256,
        cnn_layers: int = 3,
        token_dim: int = 256,
        stride: int = 25,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        embed_dim: int = 512,
        text_width: int = 768,
        dropout: float = 0.1,
        max_tokens: int = 2048,
    ):
        super().__init__()
        self.morphology = MorphologyEncoder(
            in_channels=in_channels,
            hidden_dim=cnn_hidden_dim,
            num_layers=cnn_layers,
        )
        self.tokenizer = TemporalTokenizer(
            in_dim=cnn_hidden_dim,
            token_dim=token_dim,
            stride=stride,
        )
        self.register_buffer("pos_embed", get_sinusoidal_pos_embed(max_tokens, token_dim))
        self.rhythm = RhythmTransformer(
            embed_dim=token_dim,
            num_heads=transformer_heads,
            depth=transformer_depth,
            dropout=dropout,
        )
        self.pool = AttentionPooling(token_dim)
        self.global_proj = nn.Linear(token_dim, embed_dim)
        self.token_proj = nn.Linear(token_dim, text_width)
        self.global_norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.global_proj.weight)
        nn.init.xavier_uniform_(self.token_proj.weight)
        if self.global_proj.bias is not None:
            nn.init.zeros_(self.global_proj.bias)
        if self.token_proj.bias is not None:
            nn.init.zeros_(self.token_proj.bias)

    def set_grad_checkpointing(self, enable: bool = True):
        """
        Compatibility hook for OpenCLIP/CoCa.
        ECGHybridEncoder does not currently implement activation checkpointing.
        """
        return

    def lock(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False):
        module_groups = [
            self.morphology,
            self.tokenizer,
            self.rhythm,
            self.pool,
            self.global_proj,
            self.token_proj,
            self.global_norm,
        ]

        for module in module_groups:
            for p in module.parameters():
                p.requires_grad = False

        if unlocked_groups > 0:
            for module in module_groups[-unlocked_groups:]:
                for p in module.parameters():
                    p.requires_grad = True

        if freeze_bn_stats:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def lock_for_caption_token_level(self, freeze_bn_stats: bool = True):
        """
        Caption-stage ECG tuning:
        - freeze morphology, tokenizer, pooling, global projection, global norm
        - unfreeze rhythm transformer and token projection
        """
        for module in [
            self.morphology,
            self.tokenizer,
            self.rhythm,
            self.pool,
            self.global_proj,
            self.token_proj,
            self.global_norm,
        ]:
            for p in module.parameters():
                p.requires_grad = False

        for p in self.rhythm.parameters():
            p.requires_grad = True
        for p in self.token_proj.parameters():
            p.requires_grad = True

        if freeze_bn_stats:
            for m in self.morphology.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, 12, T], e.g. [B, 12, 5000]
        returns:
            ecg_embedding: [B, embed_dim]
            ecg_tokens: [B, N, text_width]
        """
        x = self.morphology(x)
        x = self.tokenizer(x)
        n = x.size(1)
        x = x + self.pos_embed[:, :n, :].to(device=x.device, dtype=x.dtype)
        x = self.rhythm(x)

        ecg_tokens = self.token_proj(x)
        pooled = self.pool(x)
        ecg_embedding = self.global_proj(pooled)
        ecg_embedding = self.global_norm(ecg_embedding)

        return ecg_embedding, ecg_tokens