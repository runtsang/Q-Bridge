import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class AdvancedSamplerQNN(nn.Module):
    """
    A richer classical sampler network.
    Architecture:
        Linear(2 -> 8) -> BatchNorm1d -> Dropout(0.2) -> Tanh
        Linear(8 -> 4) -> BatchNorm1d -> Dropout(0.2) -> Tanh
        Linear(4 -> 2) -> Softmax
    Provides a `sample` method returning discrete samples.
    """
    def __init__(self, dropout: float = 0.2, seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.BatchNorm1d(8),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning a probability distribution over 2 outcomes.
        """
        return self.net(inputs)

    def sample(self, inputs: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the network output.
        :param inputs: Tensor of shape (..., 2)
        :param n_samples: Number of samples to draw per input
        :return: Tensor of shape (..., n_samples) with integer labels 0 or 1
        """
        probs = self.forward(inputs)
        categorical = Categorical(probs)
        return categorical.sample((n_samples,)).transpose(0, -1)

__all__ = ["AdvancedSamplerQNN"]
