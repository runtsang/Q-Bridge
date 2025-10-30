from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, Tuple

# --------------------------------------------------------------------------- #
# Classical sub‑blocks
# --------------------------------------------------------------------------- #

class ClassicalSelfAttention:
    """Simple self‑attention layer using torch tensors."""
    def __init__(self, embed_dim: int = 4):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class FullyConnectedLayer(nn.Module):
    """A tiny fully‑connected layer with a single output."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()


@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Light‑weight MLP auto‑encoder."""
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


class ConvFilter(nn.Module):
    """2‑D convolutional filter emulating a quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


# --------------------------------------------------------------------------- #
# Hybrid wrapper
# --------------------------------------------------------------------------- #

class HybridSelfAttention:
    """Combines classical attention, FC, auto‑encoder and conv blocks."""
    def __init__(
        self,
        embed_dim: int = 4,
        n_features: int = 1,
        input_dim: int = 16,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        kernel_size: int = 2,
        threshold: float = 0.0,
    ) -> None:
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        self.fcl = FullyConnectedLayer(n_features=n_features)
        self.autoencoder = AutoencoderNet(
            AutoencoderConfig(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        )
        self.conv = ConvFilter(kernel_size=kernel_size, threshold=threshold)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        fcl_thetas: Iterable[float],
    ) -> dict[str, np.ndarray]:
        """Execute all sub‑modules and return a dictionary of results."""
        attn_out = self.attention.run(rotation_params, entangle_params, inputs)
        fcl_out = self.fcl.run(fcl_thetas)
        latent = self.autoencoder.encode(torch.as_tensor(attn_out, dtype=torch.float32))
        recon = self.autoencoder.decode(latent).detach().numpy()
        conv_out = self.conv.run(attn_out[: self.conv.kernel_size ** 2])
        return {
            "attention": attn_out,
            "fcl": fcl_out,
            "reconstruction": recon,
            "conv": conv_out,
        }


def SelfAttention() -> HybridSelfAttention:
    """Factory matching the original interface."""
    return HybridSelfAttention()

__all__ = ["SelfAttention", "HybridSelfAttention"]
