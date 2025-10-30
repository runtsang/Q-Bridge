"""Hybrid classical‑quantum fully‑connected layer.

The module defines a *classical* implementation that mirrors the quantum
counterpart while exposing the same public API.  It combines:
  • An auto‑encoder that learns a compact latent representation.
  • A radial‑basis‑function kernel that can be swapped with a quantum
    kernel implementation.
  • A lightweight sampler neural network that interprets the linear
    output as a probability distribution.

The public class `FCL` accepts a ``mode`` argument; ``classical`` uses the
autoencoder+kernel+sampler stack, whereas ``quantum`` expects a quantum
implementation of the kernel and sampler to be provided externally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


# --------------------------------------------------------------------------- #
# Auto‑encoder utilities
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron auto‑encoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
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


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)


# --------------------------------------------------------------------------- #
# Kernel utilities
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """Classical RBF kernel implementation, kept for API compatibility."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """RBF kernel module that wraps :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: List[torch.Tensor], b: List[torch.Tensor], gamma: float = 1.0) -> torch.Tensor:
    """Compute a Gram matrix between two lists of tensors."""
    kernel = Kernel(gamma)
    return torch.stack([torch.stack([kernel(x, y) for y in b]) for x in a])


# --------------------------------------------------------------------------- #
# Sampler neural‑network
# --------------------------------------------------------------------------- #
class SamplerQNN(nn.Module):
    """Lightweight classifier that turns a scalar into a 2‑class probability."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


# --------------------------------------------------------------------------- #
# Hybrid fully‑connected layer
# --------------------------------------------------------------------------- #
class FCL(nn.Module):
    """
    Hybrid fully‑connected layer that can operate in either classical or
    quantum mode.  The public API is identical to the original anchor
    implementation – a ``run`` method that accepts an iterable of floats
    and returns a NumPy array.

    In classical mode the flow is::

        input → AutoEncoder → Linear → SamplerQNN

    In quantum mode the flow is::

        input → QuantumKernel (w.r.t. a fixed reference set)
              → Linear → SamplerQNN

    The quantum kernel can be swapped with a real quantum implementation
    by providing a callable that returns a torch.Tensor of similarity
    values.
    """
    def __init__(
        self,
        n_features: int = 1,
        mode: str = "classical",
        *,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        gamma: float = 1.0,
        reference_vectors: List[torch.Tensor] | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        self.device = device or torch.device("cpu")
        self.linear = nn.Linear(1, 1).to(self.device)

        if self.mode == "classical":
            self.pre = Autoencoder(
                input_dim=n_features,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            ).to(self.device)
            self.kernel = None
        else:
            # In quantum mode we pre‑compute a set of reference vectors
            # that will be used to evaluate the quantum kernel.
            if reference_vectors is None:
                # initialise with random unit vectors
                rng = torch.Generator().manual_seed(42)
                reference_vectors = [
                    torch.randn(n_features, generator=rng).to(self.device)
                    for _ in range(5)
                ]
            self.references = reference_vectors
            # The kernel is a thin wrapper that calls the external quantum
            # implementation.  For the classical fallback we use the RBF
            # kernel defined above.
            self.kernel = Kernel(gamma).to(self.device)

        self.sampler = SamplerQNN().to(self.device)

    def _process_classical(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through auto‑encoder, squeeze to a scalar
        latent = self.pre.encode(x)
        # Use the first latent dimension as the scalar feature
        feature = latent[:, 0].unsqueeze(-1)
        return feature

    def _process_quantum(self, x: torch.Tensor) -> torch.Tensor:
        # Compute similarity to each reference vector
        sims = torch.stack([self.kernel(x, r) for r in self.references], dim=1)
        # Reduce to a single scalar via mean (could be replaced by a learnable layer)
        return sims.mean(dim=1, keepdim=True)

    def forward(self, inputs: Iterable[float]) -> torch.Tensor:  # type: ignore[override]
        x = _as_tensor(inputs).to(self.device)
        if self.mode == "classical":
            feature = self._process_classical(x)
        else:
            feature = self._process_quantum(x)
        lin_out = torch.tanh(self.linear(feature))
        prob = self.sampler(lin_out)
        return prob

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Compatibility wrapper that mimics the original anchor API."""
        prob = self.forward(thetas)
        return prob.detach().cpu().numpy()


__all__ = ["FCL", "Autoencoder", "AutoencoderNet", "AutoencoderConfig",
           "Kernel", "KernalAnsatz", "kernel_matrix", "SamplerQNN"]
