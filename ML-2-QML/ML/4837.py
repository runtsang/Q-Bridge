"""Hybrid classical regression model combining a CNN feature extractor, an RBF kernel, and a linear head.

The module is fully classical, leveraging PyTorch and NumPy.  It mirrors the structure of the original
QuantumRegression anchor while incorporating the convolutional backbone from QuantumNAT and the
radial‑basis kernel utilities from QuantumKernelMethod.  The dataset generator remains unchanged,
producing sinusoidally‑modulated superposition labels from uniformly sampled unit‑cube features.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of superposition states and corresponding labels.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset wrapper around the superposition generator.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class KernalAnsatz(nn.Module):
    """
    Simple RBF kernel ansatz providing a differentiable similarity measure.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """
    Wrapper for KernalAnsatz that ensures input tensors are flattened.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


class QModel(nn.Module):
    """
    Classical regression model that extracts features with a CNN, optionally
    leverages a kernel layer for similarity regularisation, and projects to a scalar output.
    """
    def __init__(self, num_features: int = 4, use_kernel: bool = False, gamma: float = 1.0) -> None:
        super().__init__()
        self.use_kernel = use_kernel
        # CNN backbone inspired by QuantumNAT
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flattened feature size is 16 * 7 * 7 for 28x28 input
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)
        if self.use_kernel:
            self.kernel = Kernel(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)
        flattened = feat.view(bsz, -1)
        out = self.fc(flattened)
        out = self.norm(out)
        if self.use_kernel:
            # Example of using the kernel as a regulariser: compute similarity with a random batch
            random_batch = torch.randn_like(out)
            out = out + 0.01 * self.kernel(out, random_batch)
        return out.squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data", "Kernel", "KernalAnsatz"]
