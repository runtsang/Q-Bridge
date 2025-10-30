"""Enhanced classical sampler network with residual connections and dropout."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNNGen(nn.Module):
    """
    Two‑qubit sampler implemented in PyTorch.

    Architecture:
      - 3‑layer MLP with BatchNorm, ReLU, and Dropout.
      - Residual skip from input to output.
      - Softmax output producing a probability distribution over two classes.
      - ``sample`` method draws discrete samples via the Gumbel‑max trick.
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 8,
        out_features: int = 2,
        dropout: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
        )
        self.residual = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a probability distribution over the output classes.
        """
        out = self.net(x)
        res = self.residual(x)
        out = out + res
        out = self.dropout(out)
        return F.softmax(out, dim=-1)

    def log_prob(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the log‑likelihood of the target under the output distribution.
        """
        probs = self.forward(x)
        return torch.log(
            probs.gather(1, target.unsqueeze(-1)).clamp_min(1e-9)
        )

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw discrete samples from the output distribution using the Gumbel‑max trick.
        """
        probs = self.forward(x)
        gumbel = -torch.log(-torch.log(torch.rand_like(probs)))
        logits = torch.log(probs + 1e-9) + gumbel
        return (
            torch.argmax(logits, dim=-1)
           .unsqueeze(-1)
           .repeat(1, num_samples)
           .flatten()
        )
