import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math

class SamplerQNN(nn.Module):
    """
    Classical sampler network with configurable depth, dropout and
    log‑probability support.  The network is inspired by the original
    2‑to‑2 architecture but generalises to arbitrary input size.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 2,
                 dropout: float = 0.1, use_layer_norm: bool = True) -> None:
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return a probability vector over the output classes.
        """
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def log_prob(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the log‑likelihood of `target` given the network output.
        """
        probs = self.forward(x)
        return torch.log(probs.gather(-1, target.unsqueeze(-1))).squeeze(-1)

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the
        network output.  Supports batched input.
        """
        probs = self.forward(x)
        dist = Categorical(probs)
        return dist.sample((num_samples,)).transpose(0, 1)

    def loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Negative log‑likelihood loss.
        """
        return -self.log_prob(x, target).mean()

__all__ = ["SamplerQNN"]
