"""Hybrid classical neural network estimator with fast evaluation utilities.

This module extends the original EstimatorQNN example by:
- Adding a fully‑connected layer abstraction (FCL) that can be reused
  as a lightweight surrogate for quantum layers.
- Providing a FastBaseEstimator that evaluates the network on batches
  of parameter sets with optional Gaussian shot noise, mirroring the
  quantum fast‑estimator API.
- Exposing a single EstimatorQNN class that can be instantiated with
  a custom architecture or the default 2→8→4→1 network.
"""

import torch
from torch import nn
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

# --------------------------------------------------------------------------- #
# Utility: Fully‑connected surrogate layer
# --------------------------------------------------------------------------- #
class FCL(nn.Module):
    """A tiny linear layer that mimics the quantum FCL example.

    Parameters
    ----------
    n_features : int, default=1
        Number of input features.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Forward pass exposing a run interface."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(values)).mean(dim=0).detach().numpy()

# --------------------------------------------------------------------------- #
# Core network
# --------------------------------------------------------------------------- #
class EstimatorQNN(nn.Module):
    """Classical feed‑forward regressor that can be evaluated by FastBaseEstimator.

    By default it uses the architecture from the original EstimatorQNN example
    (2 → 8 → 4 → 1).  The constructor accepts an arbitrary ``nn.Module`` so
    users can plug in custom layers – e.g. the FCL defined above.
    """
    def __init__(self, model: nn.Module | None = None) -> None:
        super().__init__()
        self.model = model if model is not None else self._default_model()

    def _default_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(inputs)

# --------------------------------------------------------------------------- #
# Fast evaluation utilities
# --------------------------------------------------------------------------- #
class _Batchify:
    """Ensure 2‑D batch tensor for a sequence of parameter sets."""
    @staticmethod
    def ensure(values: Sequence[float]) -> torch.Tensor:
        t = torch.as_tensor(values, dtype=torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t

class FastBaseEstimator:
    """Evaluate a classical network for many parameter sets and observables.

    The API mirrors the quantum FastBaseEstimator so that the same
    training loop can be used for either backend.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _Batchify.ensure(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    row.append(float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val))
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic predictions."""
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
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
            noisy.append([float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row])
        return noisy

__all__ = ["EstimatorQNN", "FastBaseEstimator", "FastEstimator", "FCL"]
