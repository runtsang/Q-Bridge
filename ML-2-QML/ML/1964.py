"""Advanced classical sampler network with configurable architecture and training utilities.

The module defines :class:`SamplerQNN`, a PyTorch neural network that outputs a categorical
distribution over two classes.  It supports arbitrary hidden layers, dropout, and
provides helper methods for sampling, computing log‑probabilities, and KL divergence
between two samplers.  The implementation is intentionally lightweight so that it can be
instantiated and trained inside larger workflows or as a drop‑in replacement for the
original seed module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Callable, Iterable


class SamplerQNN(nn.Module):
    """
    A fully‑connected network that produces a probability vector over :math:`\\mathbb{R}^2`.

    Parameters
    ----------
    input_dim : int
        Size of the input vector (default ``2``).
    hidden_sizes : Sequence[int]
        Number of units in each hidden layer (default ``[4]``).
    output_dim : int
        Number of output classes (default ``2``).
    activation : Callable[[nn.Module], nn.Module]
        Activation function applied after each hidden layer (default :class:`nn.Tanh`).
    dropout : float
        Drop‑out probability applied after each hidden layer (default ``0.0``).
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_sizes: Sequence[int] = (4,),
        output_dim: int = 2,
        activation: Callable[[nn.Module], nn.Module] = nn.Tanh,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        prev_dim = input_dim
        for hs in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_dim, hs),
                    activation(),
                ]
            )
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hs
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a probability vector (softmax over the last dimension)."""
        return F.softmax(self.net(x), dim=-1)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Draw samples from the categorical distribution defined by the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(..., input_dim)``.
        num_samples : int
            Number of independent draws per input.

        Returns
        -------
        torch.Tensor
            Samples of shape ``(..., num_samples)`` with integer values 0 or 1.
        """
        probs = self.forward(x)
        # ``multinomial`` expects probabilities over the last dimension
        # and returns indices of sampled categories.
        return torch.multinomial(probs, num_samples, replacement=True).squeeze(-1)

    def log_prob(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Log‑probability of ``target`` under the distribution defined by the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(..., input_dim)``.
        target : torch.Tensor
            Integer tensor of shape ``(...,)`` with values in ``{0, 1}``.

        Returns
        -------
        torch.Tensor
            Log‑probabilities of shape ``(...,)``.
        """
        probs = self.forward(x)
        # Gather the probability of the target class
        target = target.long()
        return torch.log(probs.gather(-1, target.unsqueeze(-1)).squeeze(-1))

    @staticmethod
    def kl_divergence(
        p: "SamplerQNN",
        q: "SamplerQNN",
        x: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence D_{KL}(p || q) over a batch of inputs ``x``."""
        p_logp = p.log_prob(x, p.sample(x))
        q_probs = q.forward(x)
        p_probs = p.forward(x)
        # Avoid log(0) by adding a tiny epsilon
        eps = 1e-12
        return torch.sum(p_probs * (torch.log(p_probs + eps) - torch.log(q_probs + eps)), dim=-1)

    def train_step(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss,
    ) -> torch.Tensor:
        """Perform a single gradient update step and return the loss."""
        self.train()
        optimizer.zero_grad()
        loss = loss_fn()(self.forward(x), target)
        loss.backward()
        optimizer.step()
        return loss.detach()
