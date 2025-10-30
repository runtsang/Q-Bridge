"""Hybrid classical network combining CNN, QCNN, and RBF‑kernel layers.

The architecture is inspired by the original QuantumNAT, QCNN, and
QuantumKernelMethod examples.  It first extracts local features with a
small convolutional backbone, then applies a sequence of
fully‑connected layers that emulate the QCNN pooling and convolution
operations, and finally projects the representation through a
trainable RBF kernel layer before a sigmoid output.

The model is fully PyTorch‑compatible and can be used as a drop‑in
replacement for the original QFCModel in downstream training scripts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBFKernelLayer(nn.Module):
    """Trainable RBF kernel layer.

    Parameters
    ----------
    in_features : int
        Dimensionality of input vectors.
    out_features : int
        Number of kernel centers / output dimensions.
    gamma : float, default=1.0
        Kernel width hyper‑parameter.
    """
    def __init__(self, in_features: int, out_features: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.randn(out_features, in_features))
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)          # [B, K, D]
        dist_sq = torch.sum(diff * diff, dim=2)                     # [B, K]
        return torch.exp(-self.gamma * dist_sq)                     # [B, K]


class HybridQuantumNAT(nn.Module):
    """Hybrid classical network that mimics the Quantum‑NAT architecture.

    The network consists of:
      * a shallow CNN feature extractor,
      * a QCNN‑style stack of linear layers performing convolution and pooling,
      * a trainable RBF kernel layer,
      * a final sigmoid output.

    The design allows the model to capture both local and global structure
    while keeping the computation classical, which serves as a baseline
    for comparison with the quantum counterpart.
    """
    def __init__(self) -> None:
        super().__init__()

        # --------------- CNN feature extractor -----------------
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 14x14
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 7x7
        )

        # --------------- QCNN‑style fully‑connected stack ----
        self.qcnn_block = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 48), nn.Tanh(),   # pool1: 48
            nn.Linear(48, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),   # pool2: 16
            nn.Linear(16, 8),  nn.Tanh()
        )

        # --------------- RBF kernel projection ---------------
        self.kernel = RBFKernelLayer(in_features=8, out_features=4, gamma=0.5)
        self.norm = nn.BatchNorm1d(4)

        # --------------- Final classifier ---------------------
        self.output = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)          # [B, 16, 7, 7]
        flat = feat.view(bsz, -1)        # [B, 784]
        qcnn_out = self.qcnn_block(flat) # [B, 8]
        kern = self.kernel(qcnn_out)     # [B, 4]
        normed = self.norm(kern)         # [B, 4]
        logits = self.output(normed)     # [B, 1]
        return torch.sigmoid(logits).squeeze(-1)


__all__ = ["HybridQuantumNAT", "RBFKernelLayer"]
