"""Hybrid classical model that extends the original QFCModel with quantum‑inspired layers.

The architecture mirrors the original QuantumNAT but replaces the quantum block with
a classical approximation that preserves the same interface. This allows seamless
switching between the classical and quantum implementations during experimentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class QFCModel(nn.Module):
    """Classical hybrid model combining CNN feature extractor with quantum‑inspired layers."""

    def __init__(self, n_wires: int = 4, use_random_quantum: bool = True) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classical encoder mimicking a quantum encoding circuit
        self.encoder = nn.Sequential(
            nn.Linear(16 * 7 * 7, n_wires),
            nn.Tanh()
        )
        # Quantum‑inspired layer: random linear + non‑linear
        if use_random_quantum:
            self.q_layer = nn.Sequential(
                nn.Linear(n_wires, n_wires),
                nn.ReLU()
            )
        else:
            self.q_layer = nn.Identity()
        # Measurement head
        self.head = nn.Linear(n_wires, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        encoded = self.encoder(flat)
        qfeat = self.q_layer(encoded)
        out = self.head(qfeat)
        return self.norm(out)

    @staticmethod
    def kernel_matrix(a: np.ndarray, b: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Classical RBF kernel matrix for use with the hybrid model."""
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        diff = a[:, None, :] - b[None, :, :]
        return np.exp(-gamma * np.sum(diff ** 2, axis=-1))


__all__ = ["QFCModel"]
