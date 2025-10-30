"""Hybrid Estimator combining autoencoding, transformer, and optional quantum head.

This module defines EstimatorQNN that provides a flexible regression
model.  The classical variant uses a fully‑connected autoencoder to
compress the input, a stack of multi‑head attention blocks, and a
linear output head.  The quantum variant is implemented in
`EstimatorQNN_qml.py` (see the companion file) and can be swapped
in by setting ``use_quantum=True`` when constructing the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Autoencoder ----------
class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

def Autoencoder(input_dim: int, *,
                latent_dim: int = 32,
                hidden_dims: tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    return AutoencoderNet(AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout))

# ---------- Transformer components ----------
class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockClassical(nn.Module):
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
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) *
                             (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# ---------- EstimatorQNN ----------
class EstimatorQNN(nn.Module):
    """Hybrid estimator that can be instantiated with a classical
    transformer backbone or a quantum‑enhanced variant (see the
    companion module EstimatorQNN_qml.py)."""

    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 num_heads: int = 4,
                 num_blocks: int = 2,
                 ffn_dim: int = 128,
                 dropout: float = 0.1,
                 use_quantum: bool = False) -> None:
        super().__init__()
        if use_quantum:
            raise RuntimeError("Quantum variant is provided in EstimatorQNN_qml.py")
        self.autoencoder = Autoencoder(input_dim, latent_dim=latent_dim)
        self.pos_encoder = PositionalEncoder(latent_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(latent_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(latent_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Autoencode and flatten
        encoded = self.autoencoder.encode(inputs)
        seq = encoded.unsqueeze(1)          # (batch, seq_len=1, embed_dim)
        seq = self.pos_encoder(seq)
        x = self.transformer(seq)
        x = self.dropout(x.mean(dim=1))
        return self.output_head(x)

__all__ = ["EstimatorQNN", "Autoencoder", "AutoencoderNet", "AutoencoderConfig"]
