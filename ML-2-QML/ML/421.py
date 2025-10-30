"""Enhanced classical sampler network with added regularisation and flexibility."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen(nn.Module):
    """
    A lightweight neural sampler that maps 2‑D inputs to a categorical
    distribution over 2 classes.  The architecture is deliberately
    generic so it can be reused in hybrid pipelines.

    Features
    --------
    * Two hidden layers with BatchNorm and ReLU
    * Dropout for regularisation
    * Configurable output dimension
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, output_dim: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Log‑softmax probabilities of shape (..., output_dim).
        """
        logits = self.net(x)
        return F.log_softmax(logits, dim=-1)
