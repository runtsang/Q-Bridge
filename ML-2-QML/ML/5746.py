"""HybridFastEstimator: lightweight batched estimator with classical and hybrid head support.

The class can evaluate a PyTorch model on a list of parameter sets, optionally add shot noise,
and allows swapping the output head between a classical sigmoid and a quantum circuit.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable, Union

class HybridHead(nn.Module):
    """Base class for a head that transforms model outputs into prediction probabilities."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class ClassicalHybridHead(HybridHead):
    """Linear + sigmoid head used for classical models."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return torch.sigmoid(logits + self.shift)

class HybridFastEstimator:
    """Fast estimator that evaluates a model (nn.Module) on batched parameters and optional shot noise."""
    def __init__(self, model: nn.Module, head: HybridHead | None = None):
        self.model = model
        self.head = head or ClassicalHybridHead(model.out_features)

    def _ensure_batch(self, values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor]] | None = None,
        shots: int | None = None,
        seed: int | None = None
    ) -> List[List[float]]:
        """Evaluate model predictions for each parameter set. Observables are optional functions applied to model outputs."""
        observables = list(observables) if observables is not None else [lambda x: x]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = self._ensure_batch(params)
                outputs = self.model(inputs)
                probs = self.head(outputs).squeeze()
                row = [float(obs(probs)) for obs in observables]
                results.append(row)
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [float(rng.normal(loc=mean, scale=max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy
        return results

class HybridQCNet(nn.Module):
    """Convolutional network with a hybrid head for binary classification."""
    def __init__(self, head: HybridHead | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = head or ClassicalHybridHead(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.drop2(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.head(x).squeeze()
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridFastEstimator", "HybridHead", "ClassicalHybridHead", "HybridQCNet"]
