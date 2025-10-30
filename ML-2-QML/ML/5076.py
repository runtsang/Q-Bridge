"""QCNNHybridModel – a hybrid classical‑quantum regression architecture."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Fast estimator utilities
# --------------------------------------------------------------------------- #
def _ensure_batch(values: list[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Evaluate a torch.nn.Module on a batch of inputs and a list of observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: list[callable] | None,
        parameter_sets: list[list[float]],
    ) -> list[list[float]]:
        if observables is None or len(observables) == 0:
            observables = [lambda outputs: outputs.mean(dim=-1)]
        results: list[list[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: list[float] = []
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
    """Wraps FastBaseEstimator and adds Gaussian shot noise."""
    def evaluate(
        self,
        observables: list[callable] | None,
        parameter_sets: list[list[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: list[list[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# --------------------------------------------------------------------------- #
# Dataset utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data with a trigonometric relationship."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """PyTorch Dataset for the synthetic regression problem."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Hybrid QCNN model
# --------------------------------------------------------------------------- #
class QCNNHybridModel(nn.Module):
    """A classical network that mimics the structure of the QCNN architecture."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 4) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(hidden_dim // 4, hidden_dim // 4), nn.Tanh())
        self.head = nn.Linear(hidden_dim // 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def evaluate(
        self,
        inputs: list[list[float]],
        observables: list[callable] | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[float]]:
        """Evaluate the network on a list of input vectors."""
        estimator = FastEstimator(self) if shots else FastBaseEstimator(self)
        return estimator.evaluate(observables, inputs, shots=shots, seed=seed)

# Factory for backward compatibility
def QCNNHybridModelFactory() -> QCNNHybridModel:
    """Return a freshly constructed QCNNHybridModel."""
    return QCNNHybridModel()

__all__ = [
    "FastBaseEstimator",
    "FastEstimator",
    "generate_superposition_data",
    "RegressionDataset",
    "QCNNHybridModel",
    "QCNNHybridModelFactory",
]
