"""Combined classical QCNN and regression model.

This module defines :class:`QCNNRegressionModel` which can operate in
two modes: ``'classify'`` and ``'regress'``.  In classification mode
it uses a lightweight QCNN‑style fully‑connected network; in
regression mode it augments the same feature extractor with an
additional linear head to predict a continuous target.  The class
also exposes utilities to generate the superposition dataset that
mirrors the quantum example and a :class:`RegressionDataset`
compatible with :class:`torch.utils.data.Dataset`.

The implementation is fully classical (PyTorch) and intentionally
avoids any quantum primitives so that it can be used in
non‑quantum environments or as a baseline for comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Dataset utilities – identical to the quantum counterpart
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a superposition‑style dataset.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input vector.
    samples : int
        Number of samples to generate.

    Returns
    -------
    x : np.ndarray
        Input features of shape ``(samples, num_features)``.
    y : np.ndarray
        Target values of shape ``(samples,)``.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """PyTorch dataset wrapping the superposition data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Classical QCNN inspired network
# --------------------------------------------------------------------------- #
class QCNNModel(nn.Module):
    """A shallow fully‑connected network that mimics the QCNN structure."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


# --------------------------------------------------------------------------- #
# Combined QCNN + regression head
# --------------------------------------------------------------------------- #
class QCNNRegressionModel(nn.Module):
    """Combines a QCNN‑style extractor with a regression head.

    Parameters
    ----------
    mode : str, optional
        Either ``'classify'`` or ``'regress'``.  The default is
        ``'classify'`` which returns a probability in ``[0, 1]``.
        ``'regress'`` returns a continuous value.
    """

    def __init__(self, mode: str = "classify") -> None:
        super().__init__()
        self.mode = mode
        self.extractor = QCNNModel()

        if mode == "regress":
            # Regression head: a simple linear layer mapping 4 features to 1
            self.reg_head = nn.Linear(4, 1)
        elif mode == "classify":
            # Classification head is already part of QCNNModel
            pass
        else:
            raise ValueError("mode must be either 'classify' or'regress'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.mode == "classify":
            # Use the built‑in sigmoid output of QCNNModel
            return self.extractor(x)
        else:
            # Regression path: reuse the internal layers up to conv3
            # and then apply the regression head
            features = self.extractor.feature_map(x)
            features = self.extractor.conv1(features)
            features = self.extractor.pool1(features)
            features = self.extractor.conv2(features)
            features = self.extractor.pool2(features)
            features = self.extractor.conv3(features)
            return self.reg_head(features).squeeze(-1)

    def set_mode(self, mode: str) -> None:
        """Switch the model between classification and regression."""
        if mode not in ("classify", "regress"):
            raise ValueError("mode must be either 'classify' or'regress'")
        self.mode = mode

__all__ = [
    "QCNNRegressionModel",
    "QCNNModel",
    "RegressionDataset",
    "generate_superposition_data",
]
