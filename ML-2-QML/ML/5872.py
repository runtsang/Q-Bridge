"""
QuanvolutionHybridClassifier – a purely classical model that mimics the structure of a quantum quanvolution layer using a Random Fourier Feature (RFF) kernel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RFFLayer(nn.Module):
    """
    Random Fourier Feature layer that approximates a shift‑invariant kernel.
    Parameters are learned via a simple linear transform followed by a tanh non‑linearity.
    """
    def __init__(self, input_dim: int, output_dim: int = 4, seed: int | None = None) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        self.W = nn.Parameter(torch.tensor(rng.normal(size=(input_dim, output_dim)), dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(rng.uniform(0, 2 * np.pi, size=(output_dim,)), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, patch_features)
        proj = x @ self.W + self.b
        return torch.tanh(proj)


class QuanvolutionHybridClassifier(nn.Module):
    """
    Classical hybrid model:
        1. 2‑D convolution to extract 2×2 patches.
        2. RFFLayer to emulate a quantum kernel on each patch.
        3. Linear classifier.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, patch_size: int = 2, stride: int = 2, seed: int | None = None) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=patch_size, stride=stride)
        # Number of patches: (28/2)^2 = 14^2 = 196
        self.rff = RFFLayer(input_dim=patch_size * patch_size, output_dim=4, seed=seed)
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, 28, 28)
        """
        patches = self.conv(x)  # shape: (batch, 1, 14, 14)
        patches = patches.view(patches.size(0), -1)  # flatten patches
        features = self.rff(patches)  # shape: (batch, 4 * 14 * 14)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybridClassifier", "RFFLayer"]
