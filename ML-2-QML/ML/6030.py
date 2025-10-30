"""Hybrid classical model that fuses convolutional feature extraction and a fully connected layer,
optionally augmented with a quantum‑inspired expectation computation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FCLQuantumHybrid(nn.Module):
    """
    A purely classical implementation of the hybrid architecture.
    The network consists of:

    1. A 2‑D convolutional front‑end (adapted from Quantum‑NAT).
    2. A fully‑connected projection to 4 output features.
    3. A tanh non‑linearity that mimics the expectation‑value form of the
       original quantum layer.

    Parameters
    ----------
    use_qc : bool, optional
        If True, the output of the linear projection is passed through
        a simple quantum‑inspired transformation.  In this classical
        module the transformation is a no‑op; it is kept for API
        compatibility with the QML variant.
    """

    def __init__(self, use_qc: bool = False) -> None:
        super().__init__()
        self.use_qc = use_qc

        # Convolutional front‑end
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )

        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 4).
        """
        # Feature extraction
        out = self.features(x)
        out = out.view(out.size(0), -1)

        # Linear projection
        out = self.fc(out)

        # Classical approximation of the quantum expectation
        out = torch.tanh(out)

        # Normalisation
        out = self.norm(out)

        return out


__all__ = ["FCLQuantumHybrid"]
