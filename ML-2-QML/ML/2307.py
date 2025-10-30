"""Hybrid estimator combining FastEstimator and QCNN architecture.

This module extends the lightweight FastEstimator framework with a QCNNModel
to provide both classical neural network evaluation and optional
quantum estimation.  The hybrid class exposes a unified `evaluate`
method that can return deterministic classical predictions and
additive Gaussian shot noise, while also allowing a quantum
estimator to be injected for hybrid experiments.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

# ---- Classical QCNN model ----
class QCNNModel(nn.Module):
    """Fully connected network mimicking the quantum convolutional
    layers of a QCNN.  It accepts 8‑dimensional input and outputs
    a single probability via a sigmoid activation."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# ---- Helper to ensure batched inputs ----
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

# ---- Base estimator (deterministic) ----
class FastBaseEstimator:
    """Base class for evaluating a PyTorch model over parameter sets."""
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

# ---- Estimator with optional shot noise ----
class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to deterministic predictions."""
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

# ---- Hybrid estimator combining FastEstimator and QCNNModel ----
class FastHybridEstimator(FastEstimator):
    """Hybrid estimator that can evaluate a classical QCNN model and
    optionally a quantum estimator.  The `evaluate` method delegates to
    the underlying FastEstimator logic, while the `evaluate_quantum`
    method accepts a quantum estimator instance that follows the
    same interface as FastBaseEstimator."""
    def __init__(self, model: nn.Module, *, quantum_estimator: Optional[FastBaseEstimator] = None) -> None:
        super().__init__(model)
        self.quantum_estimator = quantum_estimator

    def evaluate_quantum(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        if self.quantum_estimator is None:
            raise RuntimeError("No quantum estimator provided")
        return self.quantum_estimator.evaluate(observables, parameter_sets)

    def set_quantum_estimator(self, estimator: FastBaseEstimator) -> None:
        """Attach a quantum estimator for hybrid experiments."""
        self.quantum_estimator = estimator

__all__ = ["QCNNModel", "FastBaseEstimator", "FastEstimator", "FastHybridEstimator"]
