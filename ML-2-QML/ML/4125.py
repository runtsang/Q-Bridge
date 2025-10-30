"""Hybrid classical classifier/regressor with fast evaluation utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Iterable, List, Sequence, Tuple, Callable

from.FastBaseEstimator import FastEstimator

# --------------------------------------------------------------------------- #
# Data generation – inspired by the regression seed
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a superposition of |0…0⟩ and |1…1⟩ and compute a smooth target."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class SuperpositionDataset(Dataset):
    """Dataset that can be used for classification or regression."""
    def __init__(self, samples: int, num_features: int, task: str = "classify"):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        if task == "classify":
            self.labels = (self.labels > 0).astype(np.int64)
        self.task = task

    def __len__(self):  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(
                self.labels[idx],
                dtype=torch.long if self.task == "classify" else torch.float32,
            ),
        }

# --------------------------------------------------------------------------- #
# Classical model – mirrors the quantum helper interface
# --------------------------------------------------------------------------- #
class HybridQuantumClassifier(nn.Module):
    """Feed‑forward network that can act as a classifier or regressor."""
    def __init__(
        self,
        num_features: int,
        hidden_sizes: Sequence[int] = (64, 32),
        task: str = "classify",
    ):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.task = task
        self.head = nn.Linear(in_dim, 2 if task == "classify" else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return self.head(out).squeeze(-1)

    # --------------------------------------------------------------------- #
    # Evaluation API – used by FastEstimator
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute observables for a batch of input vectors."""
        self.eval()
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inp = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                out = self(inp)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results

class HybridEstimator(FastEstimator):
    """Convenience wrapper exposing the same API as the quantum estimator."""
    def __init__(self, model: nn.Module, device: str = "cpu"):
        super().__init__(model, device)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        return super().evaluate(observables, parameter_sets)

__all__ = ["HybridQuantumClassifier", "SuperpositionDataset", "HybridEstimator"]
