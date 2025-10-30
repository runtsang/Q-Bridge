"""Hybrid regression model – classical implementation.

This module defines a dataset generator, a classic MLP regression head,
and a lightweight estimator that mirrors the quantum‑side utilities.
The design follows the hybrid binary classifier’s Hybrid layer,
but for regression we use a simple linear head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Iterable, List, Callable

# --------------------------------------------------------------------------- #
# Data generation – identical to the quantum seed but with real features
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data.

    Features are uniformly distributed in [-1,1].  The target is a
    non‑linear function of the sum of the features, mirroring the quantum
    state preparation in the paired seed.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Pytorch Dataset returning feature vectors and scalar targets."""
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
# Classical hybrid head – mimics the quantum expectation layer
# --------------------------------------------------------------------------- #
class HybridRegressionModel(nn.Module):
    """Deep MLP for regression with an optional shift bias.

    The architecture is inspired by the hybrid binary classifier’s
    dense head, but the output is a single real value instead of a
    probability.  A small shift can be added to emulate the bias
    that a quantum circuit would introduce.
    """
    def __init__(self, num_features: int, hidden_dim: int = 64, shift: float = 0.0):
        super().__init__()
        self.shift = shift
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        out = self.net(state_batch)
        return (out + self.shift).squeeze(-1)  # shape: (batch,)

# --------------------------------------------------------------------------- #
# Simple estimator – matches the FastBaseEstimator pattern
# --------------------------------------------------------------------------- #
class FastEstimator:
    """Evaluate a model on a batch of inputs.

    Parameters
    ----------
    model : nn.Module
        Any ``torch.nn.Module`` compatible with ``(batch,) -> (batch,)``.
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(
        self,
        inputs: Iterable[Iterable[float]],
        *,
        device: torch.device | str = "cpu",
    ) -> List[float]:
        """Return predictions for each input vector."""
        self.model.eval()
        with torch.no_grad():
            batch = torch.as_tensor(inputs, dtype=torch.float32, device=device)
            preds = self.model(batch).cpu().numpy()
        return preds.tolist()

__all__ = ["RegressionDataset", "HybridRegressionModel", "FastEstimator"]
