import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class SamplerQNNGen094(nn.Module):
    """
    A deeper, regularised feed‑forward sampler network.
    Architecture:
        2‑input  →  Linear(2, 8) → Tanh
        → Linear(8, 8) → ReLU
        → Dropout(p=0.2)
        → Linear(8, 2) → Softmax
    The final layer outputs a probability distribution over two classes.
    The ``sample`` method draws a categorical sample from that distribution.
    """
    def __init__(self, hidden_dim: int = 8, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw categorical samples from the output distribution.
        Args:
            inputs: Tensor of shape (..., 2) representing raw features.
            num_samples: Number of independent draws per input.
        Returns:
            Tensor of shape (..., num_samples) with sampled class indices.
        """
        probs = self.forward(inputs)
        dist = Categorical(probs)
        return dist.sample((num_samples,)).permute(1, 0)

__all__ = ["SamplerQNNGen094"]
