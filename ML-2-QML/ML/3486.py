from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Sequence, Iterable, Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Produce a regression dataset where the target is a non‑linear function of a linear combination
    of the features. The function emulates a quantum superposition and is intentionally
    noisy to force the model to learn a robust representation."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """PyTorch dataset that yields feature vectors and scalar targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class RegressionModel(nn.Module):
    """Feed‑forward network with batch‑norm and dropout that maps a feature vector to a scalar."""
    def __init__(self, num_features: int, hidden_sizes: Sequence[int] = (64, 32)):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.1))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FastEstimator:
    """Wrapper that evaluates a model on a sequence of parameter sets and optionally adds Gaussian noise."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def _ensure_batch(self, values: Sequence[float]) -> torch.Tensor:
        t = torch.as_tensor(values, dtype=torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return a matrix of scalar observables for each parameter set."""
        default_obs = [lambda out: out.mean(dim=-1)]
        observables = list(observables) or default_obs
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                batch = self._ensure_batch(params)
                outputs = self.model(batch)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy.append([rng.normal(mean, max(1e-6, 1 / shots)) for mean in row])
        return noisy


__all__ = ["RegressionModel", "RegressionDataset", "FastEstimator", "generate_superposition_data"]
