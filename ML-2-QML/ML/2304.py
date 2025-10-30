"""Hybrid regression module with classical patch extraction.

The implementation follows the same API as the original seed files but
uses entirely classical operations.  It can be used as a baseline or
fallback when a quantum backend is not available.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def generate_superposition_patch_data(
    num_samples: int,
    patch_size: int = 2,
    image_size: int = 28,
    channel: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic images and a regression target.

    Images are filled with random values in ``[-1, 1]``.  The target for
    each image is computed as ``sin(sum) + 0.1*cos(2*sum)`` to mimic
    the superposition‑like function used in the quantum seed.
    """
    images = np.random.uniform(-1.0, 1.0, size=(num_samples, channel, image_size, image_size)).astype(np.float32)
    sums = images.reshape(num_samples, -1).sum(axis=1)
    labels = np.sin(sums) + 0.1 * np.cos(2 * sums)
    return images, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset returning images and regression targets."""
    def __init__(self, samples: int, patch_size: int = 2, image_size: int = 28, channel: int = 1):
        self.images, self.labels = generate_superposition_patch_data(samples, patch_size, image_size, channel)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.images)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.images[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridRegressionModel(nn.Module):
    """Classical hybrid regression model.

    The model first extracts non‑overlapping 2×2 patches using a
    ``nn.Conv2d`` layer, flattens them, and feeds the resulting
    feature vector through a small MLP.  The architecture mirrors the
    quantum version but uses entirely classical operations.
    """
    def __init__(self, patch_size: int = 2, hidden_dim: int = 64):
        super().__init__()
        # Extract 2×2 patches from a single‑channel image
        self.patch_extractor = nn.Conv2d(
            in_channels=1,
            out_channels=patch_size * patch_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        # Initialize weights to identity to match patch extraction
        self.patch_extractor.weight.data = torch.eye(patch_size * patch_size).view(
            patch_size * patch_size, 1, patch_size, patch_size
        )

        self.mlp = nn.Sequential(
            nn.Linear(14 * 14 * patch_size * patch_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of images of shape ``(B, 1, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Regression output of shape ``(B,)``.
        """
        patches = self.patch_extractor(x)  # shape (B, 4, 14, 14)
        features = patches.view(patches.size(0), -1)
        return self.mlp(features).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_patch_data"]
