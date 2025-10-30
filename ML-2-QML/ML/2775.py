"""Hybrid convolutional network integrating classical and quantum-inspired layers.

This module provides a drop-in replacement for the original Conv.py
while extending the architecture with a quantum-inspired filter and
a QCNN-style pooling sequence.  The resulting model can be used
in purely classical settings or as a reference for hybrid training.
"""

from __future__ import annotations

import torch
from torch import nn

class ClassicalConvFilter(nn.Module):
    """Standard 2‑D convolution filter using a single‑channel kernel."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        logits = self.conv(patch)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

class QuantumConvFilter(nn.Module):
    """Classical surrogate for the quantum filter used in Conv.py.

    The implementation replaces the quantum circuit with a learnable
    linear layer followed by a sigmoid activation.  It mimics the
    probabilistic output of the quantum routine while remaining
    differentiable and fully compatible with PyTorch.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.linear = nn.Linear(kernel_size * kernel_size, 1)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        flat = patch.view(patch.size(0), -1)
        logits = self.linear(flat)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

class HybridConvQCNNModel(nn.Module):
    """Hybrid model that processes image patches with both classical
    and quantum‑inspired filters and aggregates the results with a
    lightweight fully‑connected head.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 num_classes: int = 2) -> None:
        super().__init__()
        self.cls_filter = ClassicalConvFilter(kernel_size, threshold)
        self.qm_filter  = QuantumConvFilter(kernel_size, threshold)
        # The head expects two scalar features per patch.
        self.fc = nn.Linear(2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W)
        batch, _, H, W = x.shape
        k = self.cls_filter.kernel_size

        # Extract non‑overlapping patches
        patches = x.unfold(2, k, k).unfold(3, k, k)
        # patches shape: (batch, 1, nH, nW, k, k)
        nH, nW = patches.shape[2], patches.shape[3]
        patches = patches.reshape(batch * nH * nW, 1, k, k)

        # Classical filter output
        cls_out = self.cls_filter(patches).view(-1, 1)  # (batch*nH*nW, 1)

        # Quantum‑inspired filter output
        qm_out = self.qm_filter(patches).view(-1, 1)

        # Concatenate feature maps
        features = torch.cat([cls_out, qm_out], dim=1)  # (batch*nH*nW, 2)

        # Aggregate across spatial dimensions
        features = features.mean(dim=0, keepdim=True)  # (1, 2)

        logits = self.fc(features)  # (1, num_classes)
        return logits.squeeze(0)

def HybridConvQCNN() -> HybridConvQCNNModel:
    """Factory returning a ready‑to‑train HybridConvQCNNModel."""
    return HybridConvQCNNModel()

__all__ = ["HybridConvQCNN", "HybridConvQCNNModel"]
