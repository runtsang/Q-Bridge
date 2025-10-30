"""Hybrid convolutional filter combining classical stride‑2 conv and a classical quantum‑kernel emulator.

This module merges concepts from:
- Conv.py: classical stride‑2 convolution.
- Quanvolution.py: quantum kernel patch encoder.
- FCL.py: fully‑connected head.
- QuantumRegression.py: dataset and model structure.

The returned object can be used as a drop‑in replacement for the quantum filter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class _Backbone(nn.Module):
    """Lightweight stride‑2 convolution used as the classical backbone."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class _QuantumEmulation(nn.Module):
    """Classical emulation of a quantum kernel via a random projection."""
    def __init__(self, in_features: int = 4, out_features: int = 4, seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.projection = torch.randn(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        return torch.tanh(x @ self.projection)

class _HybridConvFilter(nn.Module):
    """Hybrid convolutional filter that combines a classical backbone with a quantum‑kernel emulator."""
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 4,
                 kernel_size: int = 2,
                 stride: int = 2,
                 patch_size: int = 2,
                 quantum_out: int = 4,
                 num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = _Backbone(in_channels, out_channels, kernel_size, stride)
        self.quantum = _QuantumEmulation(out_features=quantum_out)
        self.patch_size = patch_size
        self.quantum_out = quantum_out
        self.classifier = nn.Linear(out_channels * (28 // patch_size) ** 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits for 10 classes.
        """
        conv_out = self.backbone(x)  # (batch, out_channels, H', W')
        batch, c, h, w = conv_out.shape
        patches = conv_out.reshape(batch, c, h * w).permute(0, 2, 1)  # (batch, n_patches, c)
        quantum_features = self.quantum(patches)  # (batch, n_patches, quantum_out)
        features = quantum_features.reshape(batch, -1)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

def Conv() -> _HybridConvFilter:
    """Return a hybrid Conv object ready for training."""
    return _HybridConvFilter()

__all__ = ["Conv"]
