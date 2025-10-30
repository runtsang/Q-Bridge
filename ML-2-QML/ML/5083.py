"""Hybrid estimator that unifies classical neural networks, kernel methods,
autoencoders, and quantum sampling."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Iterable, List, Sequence, Callable, Any
from dataclasses import dataclass

# ------------------------------------------------------------
# Shared utilities
# ------------------------------------------------------------
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# ------------------------------------------------------------
# Autoencoder components
# ------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: tuple[int, int] = (128, 64)
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


def Autoencoder(input_dim: int, **kwargs) -> AutoencoderNet:
    return AutoencoderNet(AutoencoderConfig(input_dim, **kwargs))


# ------------------------------------------------------------
# Classical RBF kernel
# ------------------------------------------------------------
class KernalAnsatz(nn.Module):
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
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# ------------------------------------------------------------
# Classical SamplerQNN
# ------------------------------------------------------------
class SamplerQNN(nn.Module):
    """Simple neural sampler that outputs a probability distribution."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


# ------------------------------------------------------------
# Base estimator
# ------------------------------------------------------------
class FastBaseEstimator:
    """Base class evaluating either a neural network or a kernel or a sampler."""

    def __init__(self, model: nn.Module | Kernel | SamplerQNN):
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], Any]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        For a standard neural network: observables are callables applied to the output tensor.
        For Kernel: parameter_sets are two‑index tuples (x, y); observables map the kernel value to a scalar.
        For SamplerQNN: observables are functions applied to probability distributions.
        """
        observables = list(observables) or [lambda out: out.mean().item()]
        results: List[List[float]] = []

        if isinstance(self.model, Kernel):
            for x, y in parameter_sets:
                val = self.model(torch.tensor(x, dtype=torch.float32),
                                 torch.tensor(y, dtype=torch.float32))
                row = [obs(val) for obs in observables]
                results.append(row)
            return results

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inp = _ensure_batch(params)
                out = self.model(inp)
                row = [obs(out) for obs in observables]
                results.append(row)
        return results


# ------------------------------------------------------------
# Estimator with shot noise
# ------------------------------------------------------------
class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], Any]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


# ------------------------------------------------------------
# Hybrid estimator with optional autoencoder preprocessing
# ------------------------------------------------------------
class HybridEstimator(FastEstimator):
    """Wraps a base model and applies an optional autoencoder preprocessor."""

    def __init__(self, model: nn.Module | Kernel | SamplerQNN, preprocessor: AutoencoderNet | None = None):
        super().__init__(model)
        self.preprocessor = preprocessor

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], Any]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        if self.preprocessor is not None:
            encoded = [
                self.preprocessor.encode(torch.tensor(p, dtype=torch.float32)).cpu().numpy().tolist()
                for p in parameter_sets
            ]
        else:
            encoded = parameter_sets
        return super().evaluate(observables, encoded, shots=shots, seed=seed)


__all__ = [
    "Autoencoder",
    "AutoencoderNet",
    "AutoencoderConfig",
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "SamplerQNN",
    "FastBaseEstimator",
    "FastEstimator",
    "HybridEstimator",
]
