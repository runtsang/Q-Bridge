import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SamplerQNNGen212(nn.Module):
    """Extended classical sampler network.

    Features:
    - Two hidden layers with ReLU and dropout.
    - Output softmax over 2 classes.
    - Sample method to draw discrete samples.
    - kl_divergence static method for comparing distributions.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        probs = self.forward(inputs)
        dist = torch.distributions.Categorical(probs)
        return dist.sample((num_samples,)).transpose(0, 1)

    @staticmethod
    def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        p = torch.clamp(p, eps, 1.0)
        q = torch.clamp(q, eps, 1.0)
        return torch.sum(p * torch.log(p / q), dim=-1)

def SamplerQNN() -> SamplerQNNGen212:
    """Factory returning an instance of SamplerQNNGen212."""
    return SamplerQNNGen212()

__all__ = ["SamplerQNN"]
