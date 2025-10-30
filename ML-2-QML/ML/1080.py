"""Enhanced classical sampler network with regularisation and flexible architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSamplerQNN(nn.Module):
    """
    A two‑layer feed‑forward network with batch‑norm, dropout and a softmax output.
    Designed to mirror the structure of the quantum sampler while providing
    classical regularisation for improved generalisation.
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
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning a probability distribution via softmax.
        """
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

__all__ = ["EnhancedSamplerQNN"]
