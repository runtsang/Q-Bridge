"""Enhanced classical sampler network with regularisation and sampling utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedSamplerQNN(nn.Module):
    """
    A deeper, regularised sampler that mirrors the original QNN.

    The network stacks two hidden layers with batch‑norm and dropout,
    producing a probability vector over two classes.  A ``sample`` method
    draws discrete samples from the softmax output, facilitating
    hybrid experiments that require classical sampling.
    """

    def __init__(self, hidden_dim: int = 8, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return log‑softmax probabilities for downstream loss functions."""
        logits = self.net(inputs)
        return F.log_softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw discrete samples from the categorical distribution defined by
        the softmax output.

        Parameters
        ----------
        inputs : torch.Tensor
            Batch of input vectors (batch, 2).
        num_samples : int, optional
            Number of samples to draw per input, by default 1.

        Returns
        -------
        torch.Tensor
            Integer samples of shape (batch, num_samples).
        """
        probs = torch.exp(self.forward(inputs))
        dist = torch.distributions.Categorical(probs)
        return dist.sample((num_samples,)).transpose(0, 1)

    def train_on_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module = nn.NLLLoss(),
        optimizer: torch.optim.Optimizer | None = None,
    ) -> torch.Tensor:
        """
        One‑step training on a single batch.  Returns the loss value.

        This helper is useful for quick prototyping of hybrid training
        loops that interleave classical and quantum updates.
        """
        self.train()
        optimizer.zero_grad()
        log_probs = self.forward(inputs)
        loss = loss_fn(log_probs, targets)
        loss.backward()
        if optimizer is not None:
            optimizer.step()
        return loss.detach()


__all__ = ["EnhancedSamplerQNN"]
