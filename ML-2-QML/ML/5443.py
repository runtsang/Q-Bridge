"""Hybrid classical autoencoder + regressor with fast evaluation utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Callable, Sequence

# --------------------------------------------------------------------------- #
#  Classical autoencoder (adapted from Autoencoder.py)
# --------------------------------------------------------------------------- #

class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    """Simple MLP autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden),
                                   nn.ReLU(),
                                   nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()])
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden),
                                   nn.ReLU(),
                                   nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()])
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def Autoencoder(input_dim: int, *, latent_dim: int = 32,
                hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.1) -> AutoencoderNet:
    return AutoencoderNet(AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout))


# --------------------------------------------------------------------------- #
#  Classical regression head (adapted from EstimatorQNN.py)
# --------------------------------------------------------------------------- #

class EstimatorNet(nn.Module):
    """Lightweight regression network."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def Estimator() -> EstimatorNet:
    return EstimatorNet()


# --------------------------------------------------------------------------- #
#  Fast evaluation utilities (adapted from FastBaseEstimator.py)
# --------------------------------------------------------------------------- #

class FastBaseEstimator:
    """Deterministic evaluator for a PyTorch model."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row = []
                for obs in observables:
                    val = obs(outputs)
                    row.append(float(val.mean().item()) if isinstance(val, torch.Tensor) else float(val))
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to the deterministic evaluator."""
    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]],
                 *, shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
#  Hybrid estimator combining autoencoder, regression, and fast evaluation
# --------------------------------------------------------------------------- #

class HybridAutoEstimator:
    """Hybrid classical estimator that compresses inputs, regresses, and evaluates."""
    def __init__(self, input_dim: int) -> None:
        self.autoencoder = Autoencoder(input_dim)
        self.regressor = Estimator()
        self._model = nn.Sequential(self.autoencoder, self.regressor)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass through encoder → regressor."""
        return self.regressor(self.autoencoder.encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """Evaluate observables on the full model using FastBaseEstimator."""
        evaluator = FastBaseEstimator(self._model)
        return evaluator.evaluate(observables, parameter_sets)

    def evaluate_with_noise(self,
                            observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                            parameter_sets: Sequence[Sequence[float]],
                            *, shots: int | None = None,
                            seed: int | None = None) -> List[List[float]]:
        """Evaluate with optional shot noise."""
        evaluator = FastEstimator(self._model)
        return evaluator.evaluate(observables, parameter_sets, shots=shots, seed=seed)


__all__ = ["HybridAutoEstimator", "Autoencoder", "Estimator", "FastBaseEstimator", "FastEstimator"]
