"""Unified classical regression model with image‑based dataset and CNN feature extractor."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

def generate_image_data(num_samples: int, img_size: int = 28, noise_level: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic image regression dataset.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    img_size : int, default 28
        Height and width of the square images.
    noise_level : float, default 0.05
        Standard deviation of Gaussian noise added to the image.

    Returns
    -------
    images : np.ndarray
        Array of shape (num_samples, 1, img_size, img_size) with pixel values in [-1, 1].
    targets : np.ndarray
        Array of shape (num_samples,) with regression targets derived from a
        trigonometric function of two latent coordinates.
    """
    # latent coordinates
    x = np.random.uniform(-1.0, 1.0, size=(num_samples, 2)).astype(np.float32)
    # image pixel values derived from sin/cos of the latent variables
    angles = 3 * x[:, 0] + 2 * x[:, 1]
    images = np.sin(angles[:, None, None] + np.arange(img_size)[:, None] * 0.1
                    + np.arange(img_size)[None, :] * 0.1)
    images = images.reshape(num_samples, 1, img_size, img_size).astype(np.float32)
    images += np.random.normal(scale=noise_level, size=images.shape).astype(np.float32)
    images = np.clip(images, -1.0, 1.0)
    # target function
    targets = np.sin(2 * x[:, 0]) * np.cos(x[:, 1])
    return images, targets.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset returning image tensors and regression targets.

    The dataset can be used directly with a PyTorch DataLoader.
    """
    def __init__(self, samples: int, img_size: int = 28):
        self.images, self.targets = generate_image_data(samples, img_size)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.images)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "image": torch.tensor(self.images[index], dtype=torch.float32),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    Classical CNN + fully‑connected regression head.

    The architecture mirrors the quantum‑inspired QFCModel from the
    Quantum‑NAT example but operates purely on classical tensors.
    """
    def __init__(self, img_size: int = 28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flattened_size = (img_size // 4) * (img_size // 4) * 16
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.norm = nn.BatchNorm1d(1)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(image_batch)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return self.norm(x).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_image_data"]
