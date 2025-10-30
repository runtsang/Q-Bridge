"""Hybrid classical QCNN model with fast batch evaluation and optional shot noise."""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Sequence, List, Callable, Union

# Lightweight estimator utilities from the second reference
from.FastBaseEstimator import FastEstimator

# Classical QCNN architecture from the first reference
class QCNNModel(nn.Module):
    """Stack of fully‑connected layers emulating the quantum convolution steps."""
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

class QCNNHybrid(nn.Module):
    """Hybrid QCNN model that exposes a classical forward pass and a fast estimator."""
    def __init__(self, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.model = QCNNModel()
        if dropout_rate > 0.0:
            # Add dropout after each convolution block for regularisation
            self.model.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh(), nn.Dropout(dropout_rate))
            self.model.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh(), nn.Dropout(dropout_rate))
            self.model.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Dropout(dropout_rate))
        self.estimator = FastEstimator(self.model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.model(inputs)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], Union[torch.Tensor, float]]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate a batch of parameter sets using the fast estimator."""
        # Delegate to FastEstimator which automatically handles shot noise
        return self.estimator.evaluate(
            observables=observables,
            parameter_sets=parameter_sets,
            shots=shots,
            seed=seed,
        )

def QCNN() -> QCNNHybrid:
    """Factory returning a configurable hybrid QCNN model."""
    return QCNNHybrid()

__all__ = ["QCNNHybrid", "QCNN"]
