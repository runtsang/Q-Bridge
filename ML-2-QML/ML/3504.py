"""Hybrid regression combining classical neural networks with fast deterministic and noisy estimators.

This module extends the original classical regression example by adding a
FastEstimator wrapper that can inject shot noise.  It preserves the original
Dataset and model shapes while exposing a single ``HybridRegression`` interface
that can be used interchangeably in downstream experiments.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Iterable, Sequence, Callable, List

# ------------------------------------------------------------------
# Data generation and dataset
# ------------------------------------------------------------------
def generate_classical_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic regression dataset.

    The target is a noisy sinusoid of the sum of features, mirroring the
    original `generate_superposition_data` logic but in a purely classical
    setting.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class ClassicalRegressionDataset(Dataset):
    """Torch ``Dataset`` that yields feature/label pairs for regression."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_classical_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ------------------------------------------------------------------
# Model architecture
# ------------------------------------------------------------------
class ClassicalRegressionModel(nn.Module):
    """Feed‑forward network used for regression."""
    def __init__(self, num_features: int):
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

# ------------------------------------------------------------------
# Fast estimator utilities (from the second seed)
# ------------------------------------------------------------------
class FastBaseEstimator:
    """Evaluate a PyTorch model for a batch of inputs and a sequence of observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                # Ensure a batch dimension
                inputs = torch.as_tensor(params, dtype=torch.float32)
                if inputs.ndim == 1:
                    inputs = inputs.unsqueeze(0)
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

class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot‑noise to deterministic predictions."""
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

# ------------------------------------------------------------------
# Unified hybrid regression wrapper
# ------------------------------------------------------------------
class HybridRegression:
    """
    A lightweight wrapper that exposes a common interface for both
    classical and quantum regression models.  The constructor accepts a
    PyTorch ``nn.Module``; for classical models this is typically
    ``ClassicalRegressionModel``.  The wrapper offers ``fit`` and ``predict``
    methods that delegate to the underlying module, and an ``evaluate`` method
    that uses ``FastEstimator`` to compute observables with optional
    shot noise.
    """
    def __init__(self, model: nn.Module, *, opt_lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt_lr)

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        loss_fn: nn.Module | None = None,
    ) -> None:
        """Simple training loop using MSE loss."""
        loss_fn = loss_fn or nn.MSELoss()
        self.model.train()
        for _ in range(epochs):
            for batch in train_loader:
                states, target = batch["states"], batch["target"]
                self.optimizer.zero_grad()
                pred = self.model(states)
                loss = loss_fn(pred, target)
                loss.backward()
                self.optimizer.step()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(X).squeeze(-1)

    # ------------------------------------------------------------------
    # Evaluation with FastEstimator
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        estimator = FastEstimator(self.model) if shots is not None else FastBaseEstimator(self.model)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

__all__ = [
    "HybridRegression",
    "ClassicalRegressionModel",
    "ClassicalRegressionDataset",
    "generate_classical_data",
    "FastEstimator",
    "FastBaseEstimator",
]
