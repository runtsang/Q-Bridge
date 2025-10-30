import torch
from torch import nn
import numpy as np

class HybridEstimatorQNN(nn.Module):
    """
    Classical hybrid regressor that augments raw inputs with a trainable
    self‑attention block. The attention mechanism is fully differentiable
    and can be trained jointly with the downstream regression head.
    """
    def __init__(self, input_dim: int = 2, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Trainable attention parameters
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim - 1, embed_dim))

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(input_dim + embed_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass performing classical self‑attention followed by regression.

        Args:
            x: Tensor of shape (batch, input_dim).

        Returns:
            Tensor of shape (batch, 1) with the regression output.
        """
        # Self‑attention
        query = x @ self.rotation_params.t()          # (B, embed_dim)
        key   = x @ self.entangle_params.t()          # (B, embed_dim)
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        attn_out = scores @ x                        # (B, input_dim)

        # Concatenate original features with attention output
        combined = torch.cat([x, attn_out], dim=-1)

        return self.regressor(combined)

__all__ = ["HybridEstimatorQNN"]
