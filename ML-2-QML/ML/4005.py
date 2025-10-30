"""Hybrid quantum-classical kernel model for feature extraction and kernel evaluation.

This module implements a hybrid kernel that blends a classical RBF kernel
with a quantum kernel simulated on a small circuit. It also incorporates a
CNN-based feature extractor inspired by the Quantum‑NAT architecture.
The module is fully classical and can be used as a drop‑in replacement
for the original QuantumKernelMethod in a classical pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class HybridQuantumKernelModel(nn.Module):
    """
    Hybrid kernel that combines:
    1. A classical RBF kernel on raw data.
    2. A quantum kernel simulated via product of cos((theta_i - phi_i)/2).
    3. A CNN feature extractor from Quantum‑NAT.

    Parameters
    ----------
    gamma : float
        Width parameter for the RBF kernel.
    weight_raw : float
        Weight for the classical RBF kernel component.
    weight_q : float
        Weight for the quantum kernel component.
    """

    def __init__(self,
                 gamma: float = 1.0,
                 weight_raw: float = 0.5,
                 weight_q: float = 0.5) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight_raw = weight_raw
        self.weight_q = weight_q

        # Feature extractor (Quantum‑NAT CNN)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the feature embedding for a batch of images."""
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)

    def _quantum_kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Quantum kernel computed as product over qubits of cos((a_i - b_i)/2).

        Parameters
        ----------
        a, b : torch.Tensor
            Embedding vectors of shape (batch, 4).

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (batch_a, batch_b, 1).
        """
        # Compute pairwise differences
        a_exp = a.unsqueeze(1)       # (m,1,4)
        b_exp = b.unsqueeze(0)       # (1,n,4)
        diff = a_exp - b_exp         # (m,n,4)
        cos_term = torch.cos(diff / 2.0)
        return torch.prod(cos_term, dim=-1, keepdim=True)

    def _rbf_kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Classical RBF kernel on raw data.

        Parameters
        ----------
        a, b : torch.Tensor
            Raw data tensors of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (batch_a, batch_b, 1).
        """
        a_flat = a.view(a.shape[0], -1)
        b_flat = b.view(b.shape[0], -1)
        diff = a_flat.unsqueeze(1) - b_flat.unsqueeze(0)
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """
        Compute the hybrid kernel matrix between two batches of images.

        Parameters
        ----------
        x, y : torch.Tensor
            Input tensors of shape (batch, 1, 28, 28) or similar.

        Returns
        -------
        torch.Tensor
            Hybrid kernel matrix of shape (batch_x, batch_y, 1).
        """
        emb_x = self._embed(x)
        emb_y = self._embed(y)
        raw_k = self._rbf_kernel(x, y)
        q_k = self._quantum_kernel(emb_x, emb_y)
        return self.weight_raw * raw_k + self.weight_q * q_k

    def kernel_matrix(self,
                      a: Tuple[torch.Tensor,...],
                      b: Tuple[torch.Tensor,...]) -> torch.Tensor:
        """
        Convenience wrapper to compute kernel matrices for sequences of tensors.

        Parameters
        ----------
        a, b : tuples of torch.Tensor
            Sequences of tensors, each of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (len(a), len(b), 1).
        """
        mats = []
        for ai in a:
            row = []
            for bj in b:
                row.append(self.forward(ai, bj))
            mats.append(torch.cat(row, dim=1))
        return torch.cat(mats, dim=0)
