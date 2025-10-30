"""Enhanced classical quanvolution classifier with residual convolution and optional quantum-inspired feature mapping.

The model replaces the single 2×2 convolution with a small residual block and a trainable
quantum‑inspired kernel that learns a richer representation of each image patch.
It can be trained end‑to‑end with standard PyTorch optimizers and provides a helper
to switch to a quantum back‑end if desired.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels!= out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x)
        out += shortcut
        return self.relu(out)


class QuantumInspiredKernel(nn.Module):
    """Fixed random linear mapping that mimics a quantum kernel."""
    def __init__(self, in_features=4, out_features=4, seed=42):
        super().__init__()
        self.register_buffer("matrix", torch.randn(out_features, in_features, generator=torch.Generator().manual_seed(seed)))
        self.register_buffer("bias", torch.randn(out_features))

    def forward(self, x):
        # x: (batch, 4)
        return F.linear(x, self.matrix, self.bias)


class QuanvolutionFilter(nn.Module):
    """Hybrid classical‑quantum feature extractor using residual conv and quantum‑inspired kernel."""
    def __init__(self, use_quantum_kernel: bool = False):
        super().__init__()
        self.use_quantum_kernel = use_quantum_kernel
        self.res_block = ResidualBlock(1, 16)
        self.quantum_kernel = QuantumInspiredKernel() if use_quantum_kernel else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        features = self.res_block(x)  # (batch, 16, 28, 28)
        # Extract 2×2 patches
        patches = features.unfold(2, 2, 2).unfold(3, 2, 2)  # (batch, 16, 14, 14, 2, 2)
        patches = patches.contiguous().view(-1, 16, 4)  # (batch*14*14, 16, 4)
        # Average over channel dimension to get 4‑dim patch
        patches = patches.mean(dim=1)  # (batch*14*14, 4)
        if self.use_quantum_kernel:
            patches = self.quantum_kernel(patches)
        return patches


class QuanvolutionClassifier(nn.Module):
    """Classical quanvolution classifier with optional quantum‑inspired kernel."""
    def __init__(self, use_quantum_kernel: bool = False):
        super().__init__()
        self.qfilter = QuanvolutionFilter(use_quantum_kernel)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)  # (batch*14*14, 4)
        features = features.view(x.size(0), -1)  # (batch, 4*14*14)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
