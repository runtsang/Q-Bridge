import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNExtended(nn.Module):
    """
    A richer classical sampler network that extends the original 2‑4‑2 architecture.
    Adds layer normalisation, dropout, and a residual connection for improved expressivity.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, output_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.residual = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a probability distribution over the output classes.
        """
        out = self.net(x)
        skip = self.residual(x)
        out = out + skip
        return F.softmax(out, dim=-1)

__all__ = ["SamplerQNNExtended"]
