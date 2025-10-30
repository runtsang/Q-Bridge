"""Hybrid regression model combining self‑attention, quanvolution, and QCNN-inspired layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data where labels depend on a superposition of input features."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic superposition data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention block used as a feature extractor."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(x)
        k = self.key_proj(x)
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ x

class QuanvolutionFilter(nn.Module):
    """2‑D convolutional filter based on a classical 2x2 kernel."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class QCNNBlock(nn.Module):
    """Hierarchical FC layers mimicking a QCNN style reduction."""
    def __init__(self, in_features: int, hidden: int = 32):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_features, hidden), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.head   = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return torch.sigmoid(self.head(x))

class HybridRegressionModel(nn.Module):
    """Combined classical model using attention, quanvolution, and QCNN blocks."""
    def __init__(self, num_features: int = 784):
        super().__init__()
        self.attention = ClassicalSelfAttention(num_features)
        self.quanv     = QuanvolutionFilter()
        # 784 (attention) + 784 (quanvolution) = 1568 input features
        self.qcnn      = QCNNBlock(in_features=1568)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Treat input as 1‑channel 28x28 image for quanvolution
        img = x.view(-1, 1, 28, 28)
        att = self.attention(x)
        quanf = self.quanv(img)
        merged = torch.cat([att, quanf], dim=1)
        return self.qcnn(merged)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
