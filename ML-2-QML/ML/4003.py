"""Highly efficient estimator utilities built on PyTorch.

This module implements a lightweight estimator for deterministic and noisy
evaluation of neural networks, inspired by the original FastBaseEstimator
and the QuantumRegression example.  It supports batched parameter sets,
automatic batching of inputs, optional Gaussian shot noise, and
integration with simple regression datasets.

The class ``FastBaseEstimatorGen`` can be used as a drop‑in replacement
for FastBaseEstimator and FastEstimator while exposing a richer API.

Example
-------
>>> import torch
>>> from. import FastBaseEstimatorGen, RegressionDataset, QModel
>>> model = QModel(num_features=10)
>>> est = FastBaseEstimatorGen(model)
>>> dataset = RegressionDataset(samples=256, num_features=10)
>>> params = [sample["states"].tolist() for sample in dataset]
>>> obs = [lambda x: x.mean()]
>>> est.evaluate(obs, params)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of parameters into a batch tensor.

    Parameters
    ----------
    values : Sequence[float]
        1‑D sequence of parameters.

    Returns
    -------
    torch.Tensor
        Tensor of shape (1, N) ready for batch inference.
    """
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimatorGen:
    """Evaluate a PyTorch model for batched parameter sets.

    Parameters
    ----------
    model : nn.Module
        Trained neural network accepting a batch of inputs.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables that convert a model output to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            2‑D iterable of parameters for each evaluation.
        shots : int, optional
            If provided, inject Gaussian noise with variance 1/shots.
        seed : int, optional
            Random seed for reproducible noise.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                rng.normal(mean, max(1e-6, 1 / shots)) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset of synthetic regression samples.

    Features are drawn from a superposition of |0> and |1> states
    and the target is a noisy trigonometric function.
    """

    def __init__(self, samples: int, num_features: int) -> None:
        self.features, self.labels = generate_superposition_data(
            num_features, samples
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


def generate_superposition_data(
    num_features: int, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate feature–label pairs for regression."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class QModel(nn.Module):
    """Simple feed‑forward regression network."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)


__all__ = [
    "FastBaseEstimatorGen",
    "RegressionDataset",
    "QModel",
    "generate_superposition_data",
]
