"""Hybrid classical regression model with convolutional preprocessing and noise‑aware evaluation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from typing import Iterable, List, Sequence, Callable

# ----- data generation ----------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data mimicking quantum superposition.

    Returns:
        features: (samples, num_features) array of float32
        labels:   (samples,) array of float32
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# ----- dataset -------------------------------------------------------------
class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic superposition data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ----- convolutional filter ------------------------------------------------
def Conv() -> nn.Module:
    """Return a PyTorch module that emulates the quantum filter with a 2‑D conv."""
    class ConvFilter(nn.Module):
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
            super().__init__()
            k = kernel_size
            self.conv = nn.Conv2d(1, 1, kernel_size=k, bias=True)
            self.threshold = threshold

        def forward(self, data: torch.Tensor) -> torch.Tensor:
            # data: (batch, features)
            k = int(np.sqrt(data.shape[-1]))
            tensor = data.view(-1, 1, k, k)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean(dim=[1, 2, 3])
    return ConvFilter()

# ----- model ----------------------------------------------------------------
class QModel(nn.Module):
    """Hybrid regression model: conv preprocessing + feed‑forward network."""
    def __init__(self, num_features: int, hidden_dims: Sequence[int] = (64, 32)):
        super().__init__()
        self.pre = Conv()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        conv_feat = self.pre(state_batch).unsqueeze(-1)
        out = self.net(conv_feat)
        return out.squeeze(-1)

# ----- fast estimators -----------------------------------------------------
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Evaluate a PyTorch model for batches of inputs and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
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
        return results

class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to the deterministic estimator."""
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

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data",
           "FastBaseEstimator", "FastEstimator"]
