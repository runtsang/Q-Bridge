import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    Enhanced classical sampler network.

    Features:
    - Two hidden layers with 8 units each.
    - LayerNorm and dropout for regularization.
    - Residual connections between hidden layers.
    - Temperature scaling for probability output.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8,
                 dropout: float = 0.2, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = h + self.res(h)  # residual connection
        logits = self.out(h)
        probs = F.softmax(logits / self.temperature, dim=-1)
        return probs

__all__ = ["SamplerQNN"]
