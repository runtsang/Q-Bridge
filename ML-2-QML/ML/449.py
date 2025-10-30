"""SamplerQNN: Classical neural sampler with enhanced architecture and training utilities.

This module defines a SamplerQNN class extending torch.nn.Module.
It implements a two‑hidden‑layer MLP with dropout and batch‑norm,
provides a probability output via softmax, and includes convenient
methods for loss computation, training, and sampling.  The design
mirrors the original seed but adds depth and flexibility for
research experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    Classical sampler neural network.
    Architecture: 2‑input → 128 → 256 → 2 output.
    Includes batch‑norm, dropout, and L2 regularisation.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: tuple[int,...] = (128, 256),
                 dropout: float = 0.1,
                 weight_decay: float = 1e-4) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross‑entropy loss with optional L2 regularisation."""
        ce = F.cross_entropy(logits, target)
        reg = sum(p.pow(2).sum() for p in self.parameters())
        return ce + self.weight_decay * reg

    def sample(self, num_samples: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Generate samples by drawing from the categorical distribution."""
        self.eval()
        with torch.no_grad():
            logits = self.net(torch.randn(num_samples, self.net[0].in_features, device=device))
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)

__all__ = ["SamplerQNN"]
