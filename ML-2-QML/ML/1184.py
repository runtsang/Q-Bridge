"""Enhanced classical sampler network.

This module defines :class:`SamplerQNN`, a PyTorch neural network that
mirrors the original two‑layer design but adds:
* ReLU activations and dropout for regularisation.
* A ``sample`` method that draws discrete samples via Gumbel‑softmax.
* Configurable hidden size and dropout probability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """Classical sampler with optional dropout and Gumbel‑softmax sampling.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input feature vector.
    hidden_dim : int, default 8
        Size of the hidden layer.
    output_dim : int, default 2
        Number of classes to sample from.
    dropout : float, default 0.1
        Dropout probability applied after the hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        output_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(
        self,
        inputs: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> torch.Tensor:
        """Draw discrete samples using Gumbel‑softmax.

        Parameters
        ----------
        inputs : torch.Tensor
            Input feature tensor.
        temperature : float
            Temperature for the Gumbel‑softmax.
        hard : bool
            If ``True`` return one‑hot samples; otherwise return soft samples.

        Returns
        -------
        torch.Tensor
            Sampled class distribution or one‑hot vector.
        """
        probs = self.forward(inputs)
        if hard:
            # Straight‑through estimator
            soft = F.gumbel_softmax(
                torch.log(probs + 1e-9), tau=temperature, hard=True, dim=-1
            )
        else:
            soft = F.gumbel_softmax(
                torch.log(probs + 1e-9), tau=temperature, hard=False, dim=-1
            )
        return soft


__all__ = ["SamplerQNN"]
