"""Combined classical self‑attention with autoencoder, convolution, and kernel."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

# Autoencoder configuration and network
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    return AutoencoderNet(AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout))

# Convolution filter
class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: torch.Tensor) -> torch.Tensor:
        batch = data.shape[0]
        tensor = data.view(batch, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.view(batch, -1)

# Kernel
class RBFKernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * dist_sq)

# Combined self‑attention
class SelfAttentionGen240:
    """
    Classical self‑attention that internally uses a convolution filter,
    an autoencoder for dimensionality reduction, and an RBF kernel
    to compute similarity between queries and keys.
    """
    def __init__(
        self,
        embed_dim: int = 4,
        autoencoder_config: AutoencoderConfig | None = None,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
        kernel_gamma: float = 1.0,
    ) -> None:
        self.embed_dim = embed_dim
        self.autoencoder = Autoencoder(
            input_dim=conv_kernel_size * conv_kernel_size,
            latent_dim=autoencoder_config.latent_dim if autoencoder_config else 16,
            hidden_dims=autoencoder_config.hidden_dims if autoencoder_config else (64,),
            dropout=autoencoder_config.dropout if autoencoder_config else 0.0,
        )
        self.conv_filter = ConvFilter(kernel_size=conv_kernel_size, threshold=conv_threshold)
        self.kernel = RBFKernel(gamma=kernel_gamma)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32)
        conv_feats = self.conv_filter.run(inputs_t)
        latent = self.autoencoder.encode(conv_feats)
        latent_dim = latent.shape[1]
        q_proj = torch.as_tensor(rotation_params.reshape(latent_dim, self.embed_dim), dtype=torch.float32)
        k_proj = torch.as_tensor(entangle_params.reshape(latent_dim, self.embed_dim), dtype=torch.float32)
        queries = torch.matmul(latent, q_proj)
        keys = torch.matmul(latent, k_proj)
        values = latent
        scores = torch.softmax(self.kernel(queries, keys), dim=-1)
        output = torch.matmul(scores, values)
        return output.detach().cpu().numpy()

def SelfAttention() -> SelfAttentionGen240:
    cfg = AutoencoderConfig(
        input_dim=4,
        latent_dim=16,
        hidden_dims=(64,),
        dropout=0.0,
    )
    return SelfAttentionGen240(
        embed_dim=4,
        autoencoder_config=cfg,
        conv_kernel_size=2,
        conv_threshold=0.0,
        kernel_gamma=1.0,
    )

__all__ = ["SelfAttention", "SelfAttentionGen240", "Autoencoder", "AutoencoderConfig", "AutoencoderNet", "ConvFilter", "RBFKernel"]
