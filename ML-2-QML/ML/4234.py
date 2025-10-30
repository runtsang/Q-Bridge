"""Hybrid sampler integrating autoencoder, self‑attention and quantum sampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AutoencoderNet(nn.Module):
    """Lightweight MLP autoencoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int,...] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class ClassicalSelfAttention:
    """Simple self‑attention operating on NumPy arrays."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        q = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        k = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        v = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(q @ k.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ v).numpy()


class HybridSamplerQNN(nn.Module):
    """End‑to‑end model chaining autoencoder → self‑attention → quantum sampler."""

    def __init__(
        self,
        input_dim: int,
        quantum_sampler: nn.Module,
        *,
        latent_dim: int = 32,
        hidden_dims: tuple[int,...] = (128, 64),
        dropout: float = 0.1,
        attention_dim: int = 4,
    ) -> None:
        super().__init__()
        self.autoencoder = AutoencoderNet(
            input_dim, latent_dim, hidden_dims, dropout
        )
        self.attention = ClassicalSelfAttention(attention_dim)
        self.quantum_sampler = quantum_sampler

    def forward(
        self,
        x: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        # Encode to latent space
        z = self.autoencoder.encode(x)
        # Convert to NumPy for attention
        z_np = z.detach().cpu().numpy()
        # Apply self‑attention
        attended = self.attention.run(rotation_params, entangle_params, z_np)
        # Convert back to tensor
        attended_t = torch.as_tensor(attended, dtype=x.dtype, device=x.device)
        # Quantum sampling
        probs = self.quantum_sampler(attended_t)
        return probs


__all__ = ["AutoencoderNet", "ClassicalSelfAttention", "HybridSamplerQNN"]
