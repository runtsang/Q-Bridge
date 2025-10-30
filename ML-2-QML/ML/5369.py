"""Hybrid estimator combining classical PyTorch models with flexible observables and shot noise.

This module defines `HybridFastEstimator`, a lightweight wrapper around any
`torch.nn.Module`.  The estimator accepts batches of parameter vectors,
evaluates the model, and applies a list of scalar observables.  An optional
subclass `FastEstimator` adds Gaussian shot noise to the results, mirroring
the behaviour of a quantum device.

Factory helpers return the classical counterparts of the quantum seeds:
- `QFCModel` (CNN + fully‑connected projection)
- `QCNNModel` (QCNN‑style fully‑connected network)
- `FCL` (simple fully‑connected layer)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridFastEstimator:
    """Evaluate a torch.nn.Module for a list of parameter sets."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
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


class FastEstimator(HybridFastEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
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
# Factory helpers mirroring the classical seeds
# --------------------------------------------------------------------------- #

def QFCModel() -> nn.Module:
    """Return a CNN‑style model from the Quantum‑NAT seed."""
    class QFCModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Linear(16 * 7 * 7, 64),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(64, 4),
            )
            self.norm = nn.BatchNorm1d(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            bsz = x.shape[0]
            features = self.features(x)
            flattened = features.view(bsz, -1)
            out = self.fc(flattened)
            return self.norm(out)

    return QFCModel()


def QCNNModel() -> nn.Module:
    """Return a QCNN‑style fully‑connected model."""
    class QCNNModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
            self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh(), nn.Dropout(0.1))
            self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh(), nn.Dropout(0.1))
            self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh(), nn.Dropout(0.1))
            self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh(), nn.Dropout(0.1))
            self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Dropout(0.1))
            self.head = nn.Linear(4, 1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.feature_map(inputs)
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            return torch.sigmoid(self.head(x))

    return QCNNModel()


def FCL() -> nn.Module:
    """Return a simple fully‑connected layer."""
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)
            self.dropout = nn.Dropout(p=0.1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            logits = self.linear(values)
            dropped = self.dropout(logits)
            expectation = torch.tanh(dropped).mean(dim=0)
            return expectation.detach().numpy()

    return FullyConnectedLayer()


__all__ = [
    "HybridFastEstimator",
    "FastEstimator",
    "QFCModel",
    "QCNNModel",
    "FCL",
]
