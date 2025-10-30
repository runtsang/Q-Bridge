"""Enhanced classical sampler network with flexible architecture and training utilities."""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.batchnorm import BatchNorm1d

logger = logging.getLogger(__name__)

__all__ = ["SamplerQNN"]


class SamplerQNN(nn.Module):
    """
    A configurable neural sampler.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input vector.
    output_dim : int, default 2
        Number of output classes; the network outputs class probabilities.
    hidden_sizes : Iterable[int], default (4,)
        Sizes of intermediate linear layers.
    dropout_rate : float, default 0.0
        Drop‑out probability applied after each hidden layer.
    batch_norm : bool, default False
        Whether to insert a BatchNorm1d layer after each hidden linear layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 2,
        hidden_sizes: Iterable[int] | Tuple[int,...] = (4,),
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        layers = []
        in_features = input_dim

        for sz in hidden_sizes:
            layers.append(nn.Linear(in_features, sz))
            layers.append(nn.Tanh())
            if batch_norm:
                layers.append(BatchNorm1d(sz))
            if dropout_rate > 0.0:
                layers.append(Dropout(dropout_rate))
            in_features = sz

        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning log‑softmax probabilities.

        Parameters
        ----------
        inputs : torch.Tensor
            Input batch of shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Log‑softmax probabilities of shape ``(batch_size, output_dim)``.
        """
        logits = self.net(inputs)
        return F.log_softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def sample(self, log_probs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw categorical samples from the probability distribution.

        Parameters
        ----------
        log_probs : torch.Tensor
            Log‑probabilities (output of :meth:`forward`) of shape ``(batch, classes)``.
        num_samples : int
            Number of samples to draw per batch element.

        Returns
        -------
        torch.Tensor
            Integer indices of shape ``(batch, num_samples)``.
        """
        probs = log_probs.exp()
        return torch.multinomial(probs, num_samples, replacement=True)

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        label_smoothing: float = 0.0,
        ignore_index: int | None = None,
    ) -> torch.Tensor:
        """
        Cross‑entropy loss with optional label smoothing.

        Parameters
        ----------
        log_probs : torch.Tensor
            Log‑probabilities from :meth:`forward`.
        targets : torch.Tensor
            Long‑tensor of target class indices.
        label_smoothing : float
            Amount of smoothing to apply; 0 means no smoothing.
        ignore_index : int or None
            Class index to ignore during loss computation.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if label_smoothing > 0.0:
            num_classes = log_probs.size(-1)
            # Create smoothed targets: one‑hot minus smoothing
            smoothed = F.one_hot(targets, num_classes).float()
            smoothed = smoothed * (1 - label_smoothing) + label_smoothing / num_classes
            loss = -(smoothed * log_probs).sum(dim=-1)
        else:
            loss = F.nll_loss(log_probs, targets, ignore_index=ignore_index)

        return loss.mean()
