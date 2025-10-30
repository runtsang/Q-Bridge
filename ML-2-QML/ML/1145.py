import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSamplerQNN(nn.Module):
    """A richer classical sampler network with residual connections, dropout, and batch‑norm."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = out + self.residual(x)
        return F.softmax(out, dim=-1)

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        probs = self.forward(x)
        return torch.multinomial(probs, num_samples, replacement=True)

__all__ = ["EnhancedSamplerQNN"]
