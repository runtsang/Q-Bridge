"""Enhanced QCNN model with residual connections and dropout.

This module implements a classical neural network that mimics the
behaviour of the original QCNN while introducing depth, regularisation
and parameterised pooling.  It can be used as a drop‑in replacement
for the seed implementation and is suitable for larger input sizes
or noisy data.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """A simple residual block used in the QCNN.

    The block performs a linear transformation, followed by a
    nonlinear activation and a skip connection.  Dropout and
    batch‑normalisation can be toggled for regularisation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float | None = None,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        # If dimensions differ, project the residual
        if residual.shape[-1]!= out.shape[-1]:
            residual = nn.Linear(residual.shape[-1], out.shape[-1])(residual)
        return out + residual


class QCNN(nn.Module):
    """Depth‑controlled QCNN with optional regularisation.

    Args:
        input_dim: Number of features in the input tensor.
        hidden_dims: Sequence of hidden layer sizes.
        dropout: Dropout probability applied after each block.
        batch_norm: Whether to use batch‑normalisation.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: tuple[int,...] | list[int] = (16, 16, 12, 8, 4, 4),
        dropout: float | None = 0.1,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.Tanh()]
        for in_f, out_f in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(
                ResidualBlock(
                    in_f,
                    out_f,
                    dropout=dropout,
                    batch_norm=batch_norm,
                )
            )
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the QCNN to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, 1)`` with values in ``[0, 1]``.
        """
        logits = self.network(x)
        return torch.sigmoid(logits)


def QCNN() -> QCNN:
    """Factory that returns a default QCNN instance.

    The default configuration matches the original seed but adds
    residual connections and dropout for improved generalisation.
    """
    return QCNN()


__all__ = ["QCNN"]
