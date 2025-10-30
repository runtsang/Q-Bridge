from __future__ import annotations

import torch
from torch import nn
import numpy as np

class HybridFCLQCNN(nn.Module):
    """
    Classical head that consumes a vector of real‑valued quantum features
    (e.g. Pauli‑Z expectations) and produces a scalar probability.
    The architecture mirrors the feature extraction of a QCNN but
    uses only linear layers and Tanh activations for efficiency.
    """
    def __init__(self, input_dim: int = 8, hidden_dims: tuple[int, int] = (16, 8)) -> None:
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, input_dim).  Expected to be the output of the
            quantum circuit (Pauli‑Z expectation values).

        Returns
        -------
        torch.Tensor
            Shape (batch, 1).  Sigmoid output suitable for binary classification.
        """
        return torch.sigmoid(self.extractor(x))

def HybridFCLQCNNFactory() -> HybridFCLQCNN:
    """
    Factory that returns a default instance configured for 8‑qubit QCNN outputs.
    """
    return HybridFCLQCNN()

__all__ = ["HybridFCLQCNN", "HybridFCLQCNNFactory"]
