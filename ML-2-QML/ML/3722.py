"""Hybrid classical regression model.

The dataset generates 28×28 grayscale images and a continuous target
derived from the sum of pixel values.  A lightweight convolutional
back‑end extracts spatial features, followed by a linear head that
produces the regression score."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_image_regression_data(samples: int, image_size: int = 28) -> tuple[np.ndarray, np.ndarray]:
    images = np.random.rand(samples, 1, image_size, image_size).astype(np.float32)
    targets = images.sum(axis=(1, 2, 3)) + 0.1 * np.sin(images.sum(axis=(1, 2, 3)))
    return images, targets.astype(np.float32)

class HybridRegressionDataset(Dataset):
    def __init__(self, samples: int, image_size: int = 28):
        self.images, self.targets = generate_image_regression_data(samples, image_size)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.images)

    def __getitem__(self, idx: int):
        return {"image": torch.tensor(self.images[idx], dtype=torch.float32),
                "target": torch.tensor(self.targets[idx], dtype=torch.float32)}

class HybridRegressionModel(nn.Module):
    def __init__(self, image_size: int = 28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        dummy = torch.zeros(1, 1, image_size, image_size)
        with torch.no_grad():
            feat_size = self.features(dummy).shape[1]
        self.head = nn.Linear(feat_size, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.features(image)
        return self.head(feat).squeeze(-1)

__all__ = ["HybridRegressionDataset", "HybridRegressionModel"]
