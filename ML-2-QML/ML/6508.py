"""Hybrid fully‑connected layer with classical and quantum inspired components.

The module combines a lightweight CNN encoder (from Quantum‑NAT) with a
parameterized linear block that mimics the behaviour of a quantum
fully‑connected layer.  It is fully differentiable and can be used
directly in a PyTorch training loop.

The design follows the *combination* scaling paradigm: the classical
and quantum halves share the same input shape and output size, so they
can be swapped or used in tandem for ablation studies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, List


class HybridFCL(nn.Module):
    """
    Classical counterpart of the quantum hybrid layer.

    Architecture
    ------------
    * Conv2d → ReLU → MaxPool2d
    * Conv2d → ReLU → MaxPool2d
    * Flatten
    * Linear (learnable weights) → Tanh → Mean
    * Output is a single scalar per batch element.
    """

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        # Feature extractor inspired by Quantum‑NAT
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flatten size after two 2×2 poolings on 28×28 input → 7×7
        self.fc = nn.Linear(16 * 7 * 7, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classical encoder and linear layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, n_features).
        """
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        # Mimic the quantum expectation by applying tanh and averaging
        return torch.tanh(out).mean(dim=0, keepdim=True)

    def run(self, thetas: Iterable[float]) -> List[float]:
        """
        Convenience method that accepts a list of parameters and returns
        the same scalar as the forward pass, but using the supplied
        parameters instead of the learned weights.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameters to inject into the linear layer.

        Returns
        -------
        List[float]
            Scalar expectation value.
        """
        with torch.no_grad():
            theta_tensor = torch.tensor(list(thetas), dtype=torch.float32)
            # Broadcast to match batch dimension
            theta_tensor = theta_tensor.view(1, -1)
            # Use the same linear mapping but replace weights
            out = theta_tensor @ self.fc.weight.t()
            return torch.tanh(out).mean().item()


__all__ = ["HybridFCL"]
