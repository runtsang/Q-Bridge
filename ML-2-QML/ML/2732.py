"""Hybrid classical kernel with fraud‑detection feature extractor.

The class mirrors the original QuantumKernelMethod interface but augments
the radial basis function kernel with a configurable fraud‑detection
neural network.  The network is built from `FraudLayerParameters` and
acts as a feature extractor before the kernel is evaluated.  This
combination allows the user to experiment with learned embeddings
while still benefiting from the analytical RBF kernel.

Typical usage:

    from QuantumKernelMethod import QuantumKernelMethod
    feature_params = [FraudLayerParameters(...),...]
    model = QuantumKernelMethod(gamma=0.5, feature_params=feature_params)
    K = model.kernel_matrix(X, Y)
"""

from __future__ import annotations

from typing import Iterable, Sequence, Optional

import numpy as np
import torch
from torch import nn

# Import fraud‑detection utilities from the same package
from.FraudDetection import FraudLayerParameters, build_fraud_detection_program

__all__ = ["QuantumKernelMethod"]


class QuantumKernelMethod(nn.Module):
    """
    Classical RBF kernel optionally preceded by a fraud‑detection feature extractor.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        feature_params: Optional[Iterable[FraudLayerParameters]] = None,
    ) -> None:
        """
        Parameters
        ----------
        gamma : float
            RBF kernel bandwidth.
        feature_params : Iterable[FraudLayerParameters], optional
            Parameters for a sequential fraud‑detection network.  If
            provided, the network is applied to each input vector before
            the kernel is evaluated.
        """
        super().__init__()
        self.gamma = gamma
        self.feature_extractor: Optional[nn.Module] = None
        if feature_params is not None:
            # Build the fraud‑detection network
            self.feature_extractor = build_fraud_detection_program(
                input_params=next(iter(feature_params)),
                layers=feature_params,
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between two single‑sample tensors.
        """
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
            y = self.feature_extractor(y)
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """
        Return the Gram matrix for two collections of samples.
        """
        kernel = self
        return np.array([[kernel(x, y).item() for y in b] for x in a])
