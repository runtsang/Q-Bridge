"""Enhanced classical estimator with residual blocks and optional dropout."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import dropout

class EstimatorQNN(nn.Module):
    """A robust regression network that extends the original seed.

    The network stacks three fully‑connected layers with a residual
    connection from the input to the output.  Dropout is applied after
    each hidden layer when ``use_dropout`` is set to ``True``.  The
    forward pass is fully differentiable and can be used as a
    stand‑alone estimator or as the classical part of a hybrid
    model.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 8,
                 output_dim: int = 1,
                 use_dropout: bool = False,
                 dropout_prob: float = 0.1) -> None:
        """Create the residual network.

        * ``input_dim`` – the input feature size.
        * ``hidden_dim`` – number of units in each hidden layer.
        * ``output_dim`` – the regression target dimension.
        * ``use_dropout`` – whether to apply dropout after hidden layers.
        * ``dropout_prob`` – probability of dropping a unit.
        """
        super().__init__()
        self.use_dropout = use_dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.residual = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass with optional dropout and residual skip."""
        out = self.net(inputs)
        if self.use_dropout:
            out = dropout(out, p=self.dropout_prob, training=self.training)
        out = out + self.residual(inputs)
        return out

def EstimatorQNN_factory(**kwargs) -> EstimatorQNN:
    """Convenience factory that mirrors the original API."""
    return EstimatorQNN(**kwargs)

__all__ = ["EstimatorQNN", "EstimatorQNN_factory"]
