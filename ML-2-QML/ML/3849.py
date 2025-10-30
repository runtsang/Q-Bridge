"""Hybrid CNN with classical convolution followed by a quantum‑inspired filter.

The model first extracts features with a standard CNN, then applies a
classical surrogate for a quanvolution filter (a 2×2 convolution with
a sigmoid threshold).  The filter value is concatenated to the
flattened features before the final fully‑connected head.  This
architecture preserves the inductive biases of both convolutional
feature extraction and quantum filtering while remaining fully
classical and compatible with PyTorch back‑ends.

The resulting model outputs four features, matching the original
Quantum‑NAT design.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QuantumInspiredFilter(nn.Module):
    """Classical surrogate for a quanvolution filter.

    The filter is implemented as a single‑channel 2×2 convolution with
    a learnable bias.  After the convolution, a sigmoid activation
    is applied and the result is thresholded.  The returned scalar
    can be interpreted as the average probability of measuring |1>
    across a 2×2 qubit block, mirroring the behaviour of the
    Qiskit implementation in the original reference.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (B, 1, H, W)
        logits = self.conv(x)
        probs = torch.sigmoid(logits - self.threshold)
        return probs.mean(dim=[1, 2, 3])  # shape (B,)


class ConvQFCModel(nn.Module):
    """Hybrid CNN + quantum‑inspired filter + fully‑connected head.

    Architecture:
        - Conv2d(1, 8, 3, padding=1) → ReLU → MaxPool2d(2)
        - Conv2d(8, 16, 3, padding=1) → ReLU → MaxPool2d(2)
        - QuantumInspiredFilter (2×2) applied to the *original* image.
        - Concatenate the filter scalar to the flattened feature map.
        - Linear layers: 16·7·7 → 64 → 4
        - BatchNorm1d(4) for output stability.
    """
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.quantum_filter = QuantumInspiredFilter(kernel_size=2, threshold=0.0)
        # The feature map size after two 2×2 pools is 7×7 for 28×28 input.
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7 + 1, 64),  # +1 from filter scalar
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        feats = self.features(x)           # shape (B, 16, 7, 7)
        flattened = feats.view(bsz, -1)    # shape (B, 16*7*7)
        filter_val = self.quantum_filter(x)  # shape (B,)
        # Concatenate filter scalar as an additional feature
        concat = torch.cat([flattened, filter_val.unsqueeze(1)], dim=1)
        out = self.fc(concat)
        return self.norm(out)

__all__ = ["ConvQFCModel"]
