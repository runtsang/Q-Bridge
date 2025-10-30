"""Hybrid QCNN model combining classical convolution layers with a fast estimator for batch inference."""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable

class FastEstimator:
    """Evaluate model outputs with optional Gaussian shot noise."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        if shots is None:
            return results
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(torch.randn(1, generator=rng).item() * (1 / shots) + mean) for mean in row]
            noisy.append(noisy_row)
        return noisy

class QCNNHybrid(nn.Module):
    """Classical QCNN with a fast estimator for efficient batch inference."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)
        self.estimator = FastEstimator(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Delegate to the embedded FastEstimator."""
        return self.estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

def QCNN() -> QCNNHybrid:
    """Factory returning a configured QCNNHybrid model."""
    return QCNNHybrid()

__all__ = ["QCNN", "QCNNHybrid", "FastEstimator"]
