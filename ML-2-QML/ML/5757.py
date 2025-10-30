"""Hybrid fully‑connected layer with classical kernel feature map.

This module implements a hybrid neural layer that combines a radial‑basis
function kernel (from the QuantumKernelMethod reference) with a linear
read‑out.  The class can be used as a drop‑in replacement for the original
``FCL`` example, but now the output is computed from a high‑dimensional
kernel feature vector instead of a single weight.

The public API mirrors the original ``FCL`` class – it exposes a
``run(thetas, x)`` method where ``thetas`` are the linear weights applied to
the kernel features of the input ``x``.  The kernel parameters (gamma) and
the number of reference points are configurable at construction time.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable

class KernalAnsatz(nn.Module):
    """Radial‑basis kernel ansatz."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper around ``KernalAnsatz`` that normalises input shapes."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

class HybridFCLKernel(nn.Module):
    """
    Hybrid classical layer that maps an input vector ``x`` to a
    high‑dimensional kernel feature vector and then linearly combines
    those features with the supplied ``thetas``.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        n_features: int = 1,
        n_reference: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        gamma
            Width parameter of the RBF kernel.
        n_features
            Dimensionality of the input data.
        n_reference
            Number of reference vectors used to construct the feature map.
        """
        super().__init__()
        self.gamma = gamma
        self.n_features = n_features
        self.n_reference = n_reference
        # Reference points are fixed during training; they are
        # treated as trainable parameters in a full implementation.
        # Here they are random but registered as buffers so that
        # ``state_dict`` works.
        self.register_buffer("reference_x", torch.randn(n_reference, n_features))
        self.kernel = Kernel(gamma)

    def run(self, thetas: Iterable[float], x: torch.Tensor) -> np.ndarray:
        """
        Compute the hybrid output.

        Parameters
        ----------
        thetas
            Iterable of length ``n_reference`` containing the weights
            applied to the kernel features.
        x
            Input tensor of shape ``(batch, n_features)``.

        Returns
        -------
        np.ndarray
            Output of shape ``(batch,)``.
        """
        if len(thetas)!= self.n_reference:
            raise ValueError(
                f"Expected {self.n_reference} theta values, got {len(thetas)}."
            )
        # Compute kernel features between each sample and all reference points.
        # Resulting shape: (batch, n_reference)
        batch = x.shape[0]
        features = torch.zeros(batch, self.n_reference, dtype=x.dtype, device=x.device)
        for i, ref in enumerate(self.reference_x):
            features[:, i] = self.kernel(x, ref).squeeze()
        # Linear combination with supplied weights.
        weights = torch.tensor(thetas, dtype=x.dtype, device=x.device).view(
            self.n_reference, 1
        )
        out = torch.matmul(features, weights).squeeze()
        return out.detach().cpu().numpy()

__all__ = ["HybridFCLKernel"]
