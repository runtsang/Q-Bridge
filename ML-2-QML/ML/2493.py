"""Classical regression model with quanvolution-inspired feature extraction."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_image_data(num_samples: int, img_size: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic grayscale images and regression targets."""
    images = np.random.uniform(-1.0, 1.0, size=(num_samples, 1, img_size, img_size)).astype(np.float32)
    sums = images.sum(axis=(1, 2, 3))
    targets = np.sin(sums) + 0.1 * np.cos(2 * sums)
    return images, targets.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int, img_size: int = 28):
        self.images, self.targets = generate_image_data(num_samples, img_size)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.images)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "image": torch.tensor(self.images[idx], dtype=torch.float32),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


class QuanvolutionFilter(nn.Module):
    """Classical 2x2 convolution with stride 2, mimicking a quanvolution filter."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class HybridRegressionModel(nn.Module):
    """Hybrid classical regression model: quanvolution feature extractor + linear head."""
    def __init__(self, img_size: int = 28) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        out_features = 4 * (img_size // 2) * (img_size // 2)
        self.head = nn.Linear(out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        out = self.head(features)
        return out.squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_image_data"]
