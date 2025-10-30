"""Hybrid classical estimator that chains convolution, attention, autoencoder and regression.

The architecture mirrors the logic of the provided reference seeds:
  * ConvFilter – a 2×2 convolution with sigmoid activation (drop‑in quanvolution).
  * SelfAttentionLayer – single‑head self‑attention over a fixed embed dimension.
  * AutoencoderNet – lightweight MLP encoder/decoder.
  * Linear regression head – final prediction.

All components are fully differentiable and run on CPU/GPU via PyTorch.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


# ----- Convolutional filter -------------------------------------------------
class ConvFilter(nn.Module):
    """Drop‑in replacement for a quanvolution layer."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations


# ----- Self‑attention -------------------------------------------------------
class SelfAttentionLayer(nn.Module):
    """Classical self‑attention block."""

    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, seq_len, embed_dim)
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim**0.5)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


# ----- Autoencoder ---------------------------------------------------------
class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    """Simple MLP autoencoder."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ----- Hybrid estimator ----------------------------------------------------
class HybridEstimator(nn.Module):
    """
    A hybrid classical estimator that chains a convolutional filter,
    a self‑attention block, an autoencoder, and a lightweight regression head.

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Shape of a single input example: (channels, height, width).
    latent_dim : int
        Size of the autoencoder latent representation.
    """

    def __init__(self,
                 input_shape: tuple[int, int, int],
                 latent_dim: int = 32,
                 conv_kernel: int = 2,
                 attention_dim: int = 4):
        super().__init__()
        channels, height, width = input_shape
        self.conv = ConvFilter(kernel_size=conv_kernel)
        # After conv, feature map size reduces by kernel-1
        conv_out_dim = ((height - conv_kernel + 1) *
                        (width - conv_kernel + 1) *
                        channels)
        self.attention = SelfAttentionLayer(embed_dim=attention_dim)
        # Flatten conv output and feed into attention
        self.attention_fc = nn.Linear(conv_out_dim, attention_dim)
        # Autoencoder expects flattened vector
        ae_cfg = AutoencoderConfig(input_dim=attention_dim,
                                   latent_dim=latent_dim,
                                   hidden_dims=(128, 64),
                                   dropout=0.1)
        self.autoencoder = AutoencoderNet(ae_cfg)
        # Final regression head
        self.regressor = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, height, width).
        Returns
        -------
        torch.Tensor
            Regression output of shape (batch, 1).
        """
        # Convolution
        conv_out = self.conv(x)          # shape (batch, 1, H', W')
        conv_flat = conv_out.view(conv_out.size(0), -1)  # (batch, conv_out_dim)
        # Self‑attention
        attn_in = self.attention_fc(conv_flat).unsqueeze(1)  # (batch, 1, attention_dim)
        attn_out = self.attention(attn_in)  # (batch, 1, attention_dim)
        attn_out = attn_out.squeeze(1)      # (batch, attention_dim)
        # Autoencoder
        latent = self.autoencoder.encode(attn_out)  # (batch, latent_dim)
        # Regression
        out = self.regressor(latent)  # (batch, 1)
        return out

__all__ = ["HybridEstimator"]
