"""QCNNEnhanced – a residual classical analogue with dynamic activations."""

from __future__ import annotations

import torch
from torch import nn
from typing import Callable


class QCNNEnhancedModel(nn.Module):
    """
    A residual, depth‑wise feed‑forward network that mimics the quantum convolution
    hierarchy.  Each block consists of a linear layer followed by a nonlinear
    activation that is *learnable* (sigmoid, tanh, or ReLU).  The model also
    incorporates a stochastic‑depth skip connection that can be dropped during
    training, improving generalisation.
    """

    def __init__(self, in_features: int = 8, hidden_dims: list[int] | None = None,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
                 dropout_prob: float = 0.0) -> None:
        """
        Parameters
        ----------
        in_features : int, default 8
            Number of input features.
        hidden_dims : list[int], optional
            Hidden layer sizes.  The default reproduces the 8→16→12→8→4→2→1
            sequence of the seed model but with a residual connection.
        ``act_fn`` : function
            The activation function to be applied after each linear layer.
            If ``None`` defaults to ``torch.tanh``.
        dropout_prob : float
            Drop‑out probability for stochastic‑depth masking.
        """
        super().__init__()
        self.in_features = in_features
        hidden_dims = hidden_dims or [16, 12, 8, 4, 2]
        self.blocks = nn.ModuleList()
        self.activations = nn.ModuleList()
        for dim in hidden_dims:
            self.blocks.append(nn.Linear(in_features, dim))
            self.activations.append(act_fn() if act_fn else nn.Tanh())
            in_features = dim
        # Final linear head
        self.head = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional stochastic depth."""
        y = x
        for blk, act in zip(self.blocks, self.activations):
            y = act(blk(y))
            # Stochastic depth: keep the scaling factor 0‑to‑1
            if self.training and self.dropout_prob > 0.0:
                mask = torch.bernoulli(
                    torch.full(y.shape, 1.0 - self.dropout_prob, device=y.device)
                )
                y = (y * mask) / (1.0 - self.dropout_prob)
        # Residual addition
        return torch.sigmoid(self.head(y + x))

def QCNNEnhanced() -> QCNNEnhancedModel:
    """Factory that returns a fully‑configured QCNNEnhancedModel."""
    return QCNNEnhancedModel()


__all__ = ["QCNNEnhanced", "QCNNEnhancedModel"]
