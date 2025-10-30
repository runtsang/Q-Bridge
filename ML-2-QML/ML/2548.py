"""Hybrid kernel model combining CNN feature extraction with RBF kernel.

This module defines :class:`HybridKernelModel` which extends
:class:`torch.nn.Module`.  The network first extracts
spatial features with a lightweight convolutional backbone,
projects them into a low‑dimensional feature space and finally
computes a Gaussian kernel between samples.  The design is
inspired by the classical RBF kernel from
``QuantumKernelMethod`` and the CNN + FC architecture from
``QuantumNAT``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

class HybridKernelModel(nn.Module):
    """CNN + FC feature extractor with an RBF kernel.

    Parameters
    ----------
    gamma : float, optional
        Width parameter of the Gaussian kernel.  Default is 1.0.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

        # Feature extractor – mimic the two‑layer CNN of QuantumNAT
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Projection to a compact feature vector
        # 16 * 7 * 7 = 784 after two 2‑pooling stages on 28x28 input
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )

        # Normalise the final feature vector
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return normalised 4‑dimensional feature vector."""
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix between two collections of samples.

        The kernel is a Gaussian RBF applied to the 4‑dimensional
        feature vectors produced by :meth:`forward`.
        """
        self.eval()
        with torch.no_grad():
            a_feats = torch.stack([self.forward(t) for t in a])  # shape (len(a), 4)
            b_feats = torch.stack([self.forward(t) for t in b])  # shape (len(b), 4)

            diff = a_feats[:, None, :] - b_feats[None, :, :]  # (len(a), len(b), 4)
            sq_norm = torch.sum(diff * diff, dim=-1)  # (len(a), len(b))
            return np.exp(-self.gamma * sq_norm).cpu().numpy()

__all__ = ["HybridKernelModel"]
