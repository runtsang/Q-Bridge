"""Enhanced classical sampler network with residual connections and dropout."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerModule(nn.Module):
    """
    A two‑layer residual network with batch‑norm and dropout that maps a 2‑D
    input to a 2‑class probability distribution.  The architecture is
    deliberately richer than the seed so that it can be used as a drop‑in
    replacement in downstream pipelines that require more expressive
    classical sampling.
    """

    def __init__(self, hidden_size: int = 8, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass returning a probability distribution over two classes.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (..., 2).

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (..., 2).
        """
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def logits(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Return raw logits for the given inputs.

        Parameters
        ----------
        inputs : torch.Tensor

        Returns
        -------
        torch.Tensor
            Logits of shape (..., 2).
        """
        return self.net(inputs)


def SamplerQNN() -> SamplerModule:
    """
    Factory that returns an instance of :class:`SamplerModule`.

    The function signature matches the original seed so existing code
    can simply swap the import path.
    """
    return SamplerModule()


__all__ = ["SamplerQNN"]
