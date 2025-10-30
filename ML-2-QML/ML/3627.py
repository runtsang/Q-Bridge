"""Hybrid classical regression model with a CNN feature extractor.

This module implements a purely classical neural network for regression tasks
that can be used as a baseline for the quantum‑enhanced version in
`QuantumRegression__gen262_qml.py`.  The architecture is inspired by the
Quantum‑NAT CNN in the reference and is compatible with the data generated
by `generate_superposition_data` in this file.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data from a superposition state.

    The output distribution mirrors the quantum example but is purely
    classical: a linear combination of sinusoids with added noise.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

def _features_to_image(features: torch.Tensor, pixel_dim: int | None = None) -> torch.Tensor:
    """Reshape a 1‑D feature vector into a square image.

    If *pixel_dim* is None the smallest integer >= sqrt(num_features) is
    used.  Zero‑padding is applied when the number of features is not a
    perfect square.
    """
    batch, num_features = features.shape
    if pixel_dim is None:
        pixel_dim = int(np.ceil(np.sqrt(num_features)))
    padded = F.pad(features, (0, pixel_dim * pixel_dim - num_features))
    return padded.view(batch, 1, pixel_dim, pixel_dim)

class RegressionDataset(Dataset):
    """Dataset that returns a feature image and the corresponding target."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridRegressionModel(nn.Module):
    """Classical CNN followed by a fully‑connected head for regression."""

    def __init__(self, num_features: int):
        super().__init__()
        # Determine image size for reshaping
        self.pixel_dim = int(np.ceil(np.sqrt(num_features)))
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully‑connected head
        flattened = 16 * (self.pixel_dim // 4) * (self.pixel_dim // 4)
        self.head = nn.Sequential(
            nn.Linear(flattened, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.norm = nn.BatchNorm1d(1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Convert 1‑D features into a square image
        img = _features_to_image(state_batch, self.pixel_dim)
        bsz = img.shape[0]
        feats = self.features(img)
        flat = feats.view(bsz, -1)
        out = self.head(flat)
        return self.norm(out).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
