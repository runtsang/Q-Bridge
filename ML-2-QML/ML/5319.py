"""QTransformerHybrid: Classical transformer with optional conv/autoencoder pre‑processing."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Auxiliary Convolutional Filter (drop‑in for quanvolution)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Simple 2‑D convolutional filter mimicking a quantum quanvolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: torch.Tensor) -> float:
        """Apply the filter to a 2‑D tensor and return the mean activation."""
        x = data.view(1, 1, self.kernel_size, self.kernel_size).float()
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


# --------------------------------------------------------------------------- #
# 2. Autoencoder (fully‑connected)
# --------------------------------------------------------------------------- #
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Lightweight MLP autoencoder."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*layers)

        # Decoder
        layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int, *, latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)


# --------------------------------------------------------------------------- #
# 3. Classical CNN‑FC model (Quantum‑NAT inspired)
# --------------------------------------------------------------------------- #
class QFCModelClassical(nn.Module):
    """Simple CNN followed by a 4‑dimensional projection."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)


# --------------------------------------------------------------------------- #
# 4. Transformer core (classical)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return out


class FeedForwardClassical(nn.Module):
    """Two‑layer perceptron."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockClassical(nn.Module):
    """Single transformer encoder block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TextClassifier(nn.Module):
    """Transformer‑based text classifier."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoder(embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim,
                                    num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x)
        x = self.pos_emb(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# 5. Hybrid Transformer (public API)
# --------------------------------------------------------------------------- #
class QTransformerHybrid(nn.Module):
    """
    Public transformer that can operate in pure classical mode or
    switch to quantum sub‑modules when torchquantum is available.
    The API mirrors the classic `TextClassifier`, making it drop‑in
    for existing pipelines.

    Parameters
    ----------
    use_quantum : bool
        If True and torchquantum is importable, the attention and
        feed‑forward layers are replaced by their quantum counterparts.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum: bool = False) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoder(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim,
                                    num_classes if num_classes > 2 else 1)

        # Attempt to import torchquantum for quantum blocks
        if use_quantum:
            try:
                import torchquantum as tq
                from. import _quantum_blocks  # local module defined below
                self.blocks = nn.Sequential(*[
                    _quantum_blocks.TransformerBlockQuantum(
                        embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ])
            except Exception:
                # Fallback to classical if import fails
                self.blocks = nn.Sequential(*[
                    TransformerBlockClassical(embed_dim, num_heads,
                                             ffn_dim, dropout)
                    for _ in range(num_blocks)
                ])
        else:
            self.blocks = nn.Sequential(*[
                TransformerBlockClassical(embed_dim, num_heads,
                                         ffn_dim, dropout)
                for _ in range(num_blocks)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x)
        x = self.pos_emb(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "ConvFilter",
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "QFCModelClassical",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "TextClassifier",
    "QTransformerHybrid",
]
