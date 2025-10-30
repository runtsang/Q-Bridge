"""Combined QCNN model with sampler and fast estimator for efficient classical evaluation."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, List, Iterable, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class QCNNModel(nn.Module):
    """Classical QCNN-inspired network with an optional sampler head."""
    def __init__(self) -> None:
        super().__init__()
        # Feature extraction
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Head
        self.head = nn.Linear(4, 1)
        # Sampler head for probabilistic output
        self.sampler = nn.Sequential(nn.Linear(4, 2), nn.Tanh(), nn.Linear(2, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over two classes."""
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return F.softmax(self.sampler(x), dim=-1)

class SamplerQNN(nn.Module):
    """Simple probabilistic classifier used in hybrid workflows."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)

class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""
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

class FastEstimator(FastBaseEstimator):
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

class QCNNGen026(nn.Module):
    """Unified QCNN model that exposes both deterministic and sampled outputs
    and supports fast batched evaluation."""
    def __init__(self) -> None:
        super().__init__()
        self.model = QCNNModel()
        self.estimator = FastEstimator(self.model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model.sample(inputs)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        return self.estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

__all__ = [
    "QCNNGen026",
    "QCNNModel",
    "FastEstimator",
    "FastBaseEstimator",
    "SamplerQNN",
]
