"""Hybrid classical model combining a quanvolution filter, random Fourier feature mapping, and a linear head.

This module extends the original Quanvolution example by adding a quantum‑inspired feature map
using random Fourier features (RFF).  The RFF layer approximates a quantum kernel while
remaining entirely classical, enabling efficient training on CPUs or GPUs.  The module also
provides utilities to generate synthetic regression data, mirroring the QuantumRegression
seed, for quick experimentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data

__all__ = [
    "QuanvolutionHybrid",
    "QuanvolutionFilter",
    "QuantumKernelFeatureMap",
    "generate_superposition_data",
    "RegressionDataset",
]

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data mimicking a superposition state.

    The function returns a feature matrix ``x`` and a target vector ``y`` where
    ``y = sin(sum(x)) + 0.1 * cos(2 * sum(x))``.  This is inspired by the
    quantum regression example and can be used for regression or as a toy
    classification dataset.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(data.Dataset):
    """Dataset wrapping the synthetic regression data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumKernelFeatureMap(nn.Module):
    """Random Fourier feature map approximating a quantum kernel.

    The map samples random weights and biases and transforms an input vector
    ``x`` into a higher‑dimensional feature vector ``[cos(x·W + b), sin(x·W + b)]``.
    This layer is fully differentiable and can be trained jointly with the rest
    of the network.
    """

    def __init__(self, input_dim: int, output_dim: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.W = nn.Parameter(torch.randn(input_dim, output_dim) * self.scale, requires_grad=False)
        self.b = nn.Parameter(torch.randn(output_dim) * 2 * np.pi, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.W + self.b
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution with stride 2, followed by ReLU.

    The filter reduces a 28×28 image to a 14×14 feature map with 4 channels,
    mirroring the original quanvolution example but adding a non‑linearity for
    richer feature extraction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))

class QuanvolutionHybrid(nn.Module):
    """Hybrid image classifier combining classical convolution and a quantum‑inspired feature map.

    The network first applies a 2×2 convolution to extract local patches, then flattens the
    feature map and passes it through a random Fourier feature map to emulate a quantum kernel.
    Finally a linear layer produces class logits.  The model is fully differentiable
    and can be trained with standard optimizers.
    """

    def __init__(self, num_classes: int = 10, rff_dim: int = 256) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.rff = QuantumKernelFeatureMap(input_dim=4 * 14 * 14, output_dim=rff_dim)
        self.classifier = nn.Linear(rff_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        flat = features.view(features.size(0), -1)
        mapped = self.rff(flat)
        logits = self.classifier(mapped)
        return F.log_softmax(logits, dim=-1)
