"""Hybrid estimator and regression utilities for classical models.

This module extends the original FastBaseEstimator with:
* Shot‑noise support via FastEstimator.
* Regression dataset and model generation for superposition data.
* A unified interface that can be swapped with the quantum counterpart.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of parameter values to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# --------------------------------------------------------------------------- #
# Base estimator
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate a PyTorch model for a batch of parameter sets.

    Parameters
    ----------
    model
        A :class:`torch.nn.Module` that maps a batch of inputs to outputs.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute scalar observables for each parameter set.

        The default observable returns the mean of the last dimension.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


# --------------------------------------------------------------------------- #
# Shot‑noise estimator
# --------------------------------------------------------------------------- #
class FastEstimator(FastBaseEstimator):
    """Wrap :class:`FastBaseEstimator` with Gaussian shot‑noise."""

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
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
# Regression dataset
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data from a superposition of |0⟩ and |1⟩ states.

    The labels are a smooth function of the feature sum to mimic a quantum
    regression target.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """PyTorch dataset wrapping superposition data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Classical regression model
# --------------------------------------------------------------------------- #
class QModel(nn.Module):
    """Simple feed‑forward network for regression."""

    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch.to(torch.float32)).squeeze(-1)


__all__ = [
    "FastBaseEstimator",
    "FastEstimator",
    "RegressionDataset",
    "generate_superposition_data",
    "QModel",
]
