"""Enhanced feed‑forward regressor with residual blocks and regularisation.

This module extends the toy EstimatorQNN by adding:
  • Residual skip connections to facilitate gradient flow.
  • Batch‑normalisation layers for faster convergence.
  • Drop‑out regularisation to reduce over‑fitting.
  • A small helper factory to keep the public API familiar.
"""

import torch
from torch import nn

class EnhancedEstimatorQNN(nn.Module):
    """
    A light‑weight regression network that mirrors the original EstimatorQNN
    but incorporates modern best practices.

    Parameters
    ----------
    input_dim : int, default 2
        Number of scalar input features.
    hidden_dims : Sequence[int], default (16, 8)
        Sizes of the hidden layers.
    dropout : float, default 0.2
        Drop‑out probability applied after each hidden block.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: tuple[int,...] = (16, 8),
                 dropout: float = 0.2) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for idx, hdim in enumerate(hidden_dims):
            # Linear → BatchNorm → ReLU → Dropout
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.BatchNorm1d(hdim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            # Residual connection (if dimensions match)
            if prev_dim == hdim:
                layers.append(nn.Identity())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)

def EstimatorQNN() -> EnhancedEstimatorQNN:
    """
    Public factory mirror of the original EstimatorQNN function.

    Returns
    -------
    EnhancedEstimatorQNN
        A ready‑to‑use regression network.
    """
    return EnhancedEstimatorQNN()

__all__ = ["EnhancedEstimatorQNN", "EstimatorQNN"]
