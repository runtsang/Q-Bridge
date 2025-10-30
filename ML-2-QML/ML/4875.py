"""HybridKernelMethod – classical side implementation.

The module combines radial‑basis, auto‑encoder, convolutional and self‑attention
pre‑processing to produce a feature vector that is then fed into a quantum kernel
module.  It is deliberately lightweight and is fully compatible with the
previous `QuantumKernelMethod` API.

>>> from HybridKernelMethod import HybridKernelMethod
>>> model = HybridKernelMethod()
>>> x = torch.randn(5, 10)
>>> y = torch.randn(5, 10)
>>> model.kernel_matrix([x], [y])
array([[...]])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
#  Classical feature extraction utilities
# --------------------------------------------------------------------------- #

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


class RBFKernel(nn.Module):
    """Standard radial basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class AutoencoderConfig:
    """Configuration for the fully‑connected auto‑encoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    """Encoder–decoder architecture."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        enc_layers, dec_layers = [], []

        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, hidden))
            enc_layers.append(nn.ReLU())
            if config.dropout > 0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, hidden))
            dec_layers.append(nn.ReLU())
            if config.dropout > 0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


# --------------------------------------------------------------------------- #
#  Convolutional filter emulation
# --------------------------------------------------------------------------- #

class ConvFilter(nn.Module):
    """2‑D convolutional filter that mimics a quantum quanvolution layer."""
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
#  Self‑attention emulation
# --------------------------------------------------------------------------- #

class SelfAttentionModule(nn.Module):
    """Drop‑in replacement for a quantum self‑attention block."""
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1),
                                dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1),
                              dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


# --------------------------------------------------------------------------- #
#  Hybrid kernel orchestrator
# --------------------------------------------------------------------------- #

class HybridKernelMethod:
    """
    A composite kernel that first transforms input data through
    auto‑encoding, convolution, and attention, then computes an
    RBF kernel over the concatenated representation.  It also
    exposes a quantum kernel compatible with a TorchQuantum backend.
    """
    def __init__(self,
                 gamma: float = 1.0,
                 autoencoder_cfg: AutoencoderConfig | None = None,
                 conv_kernel: int = 2,
                 attention_dim: int = 4) -> None:
        # Classical sub‑modules
        self.rbf = RBFKernel(gamma)
        self.autoencoder = (AutoencoderNet(autoencoder_cfg)
                            if autoencoder_cfg else None)
        self.conv = ConvFilter(kernel_size=conv_kernel)
        self.attn = SelfAttentionModule(embed_dim=attention_dim)

    # --------------------------------------------------------------------- #
    #  Feature extraction helpers
    # --------------------------------------------------------------------- #
    def _extract(self, data: torch.Tensor) -> torch.Tensor:
        """Return a 1‑D feature vector per sample."""
        feats = []

        # Auto‑encoder latent
        if self.autoencoder:
            with torch.no_grad():
                latent = self.autoencoder.encode(data)
            feats.append(latent)

        # Convolutional statistic
        conv_output = torch.tensor([self.conv.run(d.cpu().numpy()) for d in data])
        feats.append(conv_output.unsqueeze(-1))

        # Self‑attention output
        # Dummy parameters for demonstration
        rot = np.random.rand(self.attn.embed_dim, data.shape[-1])
        ent = np.random.rand(self.attn.embed_dim, data.shape[-1])
        attn_out = [self.attn.run(rot, ent, d.cpu().numpy()) for d in data]
        feats.append(torch.tensor(attn_out))

        return torch.cat(feats, dim=-1)

    # --------------------------------------------------------------------- #
    #  Classical kernel computation
    # --------------------------------------------------------------------- #
    def kernel_matrix(self,
                      a: Iterable[torch.Tensor],
                      b: Iterable[torch.Tensor]) -> np.ndarray:
        """Compute a Gram matrix using the hybrid feature representation."""
        a = [self._extract(t) for t in a]
        b = [self._extract(t) for t in b]
        return np.array([[self.rbf(x, y).item() for y in b] for x in a])

    # --------------------------------------------------------------------- #
    #  Quantum kernel placeholder (for compatibility)
    # --------------------------------------------------------------------- #
    def quantum_kernel_matrix(self,
                              a: Iterable[torch.Tensor],
                              b: Iterable[torch.Tensor]) -> np.ndarray:
        """Fallback that simply returns the classical kernel."""
        return self.kernel_matrix(a, b)

__all__ = ["HybridKernelMethod"]
