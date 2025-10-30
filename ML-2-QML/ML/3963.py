"""Classical regression model with a residual backbone and quantum‑inspired feature engineering.

The class `QModel` implements a deep residual neural network that processes
synthetic superposition data.  It can be trained with any standard PyTorch
optimizer and serves as the purely classical counterpart to the quantum
implementation below.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Data generation                                                             #
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_features: int, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data that mimics measurements on a
    superposition of |0> and |1> states.

    Parameters
    ----------
    num_features : int
        Dimension of the input feature vector.
    samples : int
        Number of data points to generate.

    Returns
    -------
    X : np.ndarray
        Raw feature matrix of shape (samples, num_features).
    y : np.ndarray
        Regression target of shape (samples,).
    """
    X = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return X, y.astype(np.float32)


class RegressionDataset(Dataset):
    """PyTorch dataset wrapping the synthetic superposition data."""

    def __init__(self, samples: int, num_features: int):
        self.X, self.y = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.X[idx], dtype=torch.float32),
            "target": torch.tensor(self.y[idx], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Helper modules                                                             #
# --------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    """Simple residual block: Linear → BatchNorm → ReLU → Dropout → Linear,
    with a skip connection that adds the input directly to the block output."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(out_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x


# --------------------------------------------------------------------------- #
# Model definition                                                            #
# --------------------------------------------------------------------------- #
class QModel(nn.Module):
    """Hybrid regression network with a residual backbone and a quantum‑style
    head that emulates a lightweight quantum encoder using classical layers."""

    def __init__(self, num_features: int, num_wires: int = 4):
        super().__init__()
        self.backbone = nn.Sequential(
            ResidualBlock(num_features, 64),
            ResidualBlock(64, 32),
            ResidualBlock(32, 16),
        )
        # Classical head that mimics quantum feature extraction
        self.quantum_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        self.regressor = nn.Sequential(
            nn.Linear(20, 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        classical_feats = self.backbone(x)
        quantum_feats = self.quantum_head(classical_feats)
        feats = torch.cat([classical_feats, quantum_feats], dim=-1)
        return self.regressor(feats).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
