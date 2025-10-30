"""SamplerQNNExtended – a richer classical sampler network.

The module implements a two‑hidden‑layer feed‑forward network with dropout and residual
connections, providing a softmax output suitable for probability estimation.  The
class exposes a `sample` method that draws from the learned distribution, which
is useful for downstream generative tasks or for hybrid training with the quantum
counterpart.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SamplerQNNExtended(nn.Module):
    """
    A robust classical sampler network.

    Architecture
    ------------
    - Input: 2‑dimensional vector (e.g. two‑bit features).
    - Hidden layers: 4 → 8 → 4 units with Tanh activation.
    - Dropout (p=0.2) after each hidden layer to mitigate overfitting.
    - Residual connections between the first and third hidden layers.
    - Output: 2‑dimensional softmax probability vector.
    """

    def __init__(self, input_dim: int = 2, hidden_dims: Tuple[int, int, int] = (4, 8, 4), dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.Tanh(),
        )
        self.residual = nn.Linear(hidden_dims[0], hidden_dims[2])
        self.output_layer = nn.Linear(hidden_dims[2], 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a probability distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., 2).

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (..., 2).
        """
        h1 = self.net[0:3](x)          # 2 → 4
        h2 = self.net[3:6](h1)         # 4 → 8
        h3 = self.net[6:9](h2)         # 8 → 4
        # Residual addition
        h3 = h3 + self.residual(h1)
        logits = self.output_layer(h3)
        return F.softmax(logits, dim=-1)

    def sample(self, batch_size: int = 1, device: torch.device | str = "cpu") -> torch.Tensor:
        """
        Draw samples from the learned distribution.

        Parameters
        ----------
        batch_size : int
            Number of samples to draw.
        device : torch.device | str
            Device on which to perform sampling.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, 2) containing one‑hot encoded samples.
        """
        probs = self.forward(torch.zeros(batch_size, 2, device=device))
        # Categorical sampling
        return torch.multinomial(probs, num_samples=1).squeeze(-1).one_hot(2)


__all__ = ["SamplerQNNExtended"]
