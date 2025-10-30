import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SamplerQNN(nn.Module):
    """
    A richer sampler neural network that extends the original 2→4→2 architecture.
    Adds dropout, batch normalization and a configurable number of hidden layers.
    Provides a `sample` method to draw discrete samples from the output probabilities.
    """

    def __init__(self, input_dim: int = 2, hidden_dims: Tuple[int,...] = (4, 4), output_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log‑softmax probabilities."""
        return F.log_softmax(self.net(x), dim=-1)

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample discrete actions from the probability distribution produced by the network.
        Returns a tensor of shape (num_samples, batch_size) containing class indices.
        """
        probs = torch.exp(self.forward(x)).detach()
        return torch.multinomial(probs, num_samples, replacement=True)
