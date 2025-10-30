"""Hybrid regression model with attention and multi‑task output.

The classical component is a lightweight feed‑forward network that learns a
attention vector over the hidden representation of the quantum module.  The
model can be trained end‑to‑end, but the quantum part is frozen during
initial experiments to isolate its contribution.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple

def generate_superposition_data(
    num_features: int,
    samples: int,
    *,
    noise_level: float = 0.0,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a dataset of superposition states (float features).

    The function is a wrapper around the seed implementation that
    optionally adds Gaussian noise to the labels.  This allows controlled
    experiments on noise robustness.
    """
    rng = np.random.default_rng(random_state)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_level > 0.0:
        y += rng.normal(scale=noise_level, size=y.shape)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns a binary label indicating high‑frequency content.

    The binary label is 1 if |y| > 0.8 and 0 otherwise.  This encourages the
    model to learn a secondary task that can improve feature learning.
    """
    def __init__(self, samples: int, num_features: int, noise_level: float = 0.0):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, noise_level=noise_level
        )
        self.binary = (np.abs(self.labels) > 0.8).astype(np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
            "binary": torch.tensor(self.binary[index], dtype=torch.float32),
        }

class HybridQModel(nn.Module):
    """Hybrid regression/classification model with attention over hidden states.

    The architecture consists of:
      * A feed‑forward network that produces a 32‑dimensional hidden vector.
      * An attention module that learns a weight vector over the hidden
        dimensions.
      * Two heads: a regression head producing a scalar and a classifier
        head producing a binary probability.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        # Attention mechanism
        self.attn = nn.Linear(32, 32)
        # Regression head
        self.reg_head = nn.Linear(32, 1)
        # Classification head
        self.clf_head = nn.Linear(32, 1)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(states)
        # Compute attention weights
        attn_weights = torch.softmax(self.attn(hidden), dim=-1)
        weighted = hidden * attn_weights
        reg_out = self.reg_head(weighted).squeeze(-1)
        clf_out = torch.sigmoid(self.clf_head(weighted).squeeze(-1))
        return reg_out, clf_out

__all__ = ["HybridQModel", "RegressionDataset", "generate_superposition_data"]
