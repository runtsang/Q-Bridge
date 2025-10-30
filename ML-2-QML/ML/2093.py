"""Enhanced Classical QCNN architecture with dropout, residual connections, and configurable layers.

The original QCNNModel was a shallow stack of linear layers. This extension
provides a flexible architecture that can be tuned for depth, regularisation,
and non‑linearity. Users can adjust the hidden dimensions, activation function,
dropout rate, and enable residual connections to mitigate vanishing gradients.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Sequence, Callable


def _get_activation(name: str) -> Callable:
    """Map a string name to a PyTorch activation function."""
    activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


class QCNNModel(nn.Module):
    """Classical QCNN model with optional dropout, residuals, and configurable layers.

    Parameters
    ----------
    input_dim : int, default 8
        Dimension of the input features.
    hidden_dims : Sequence[int], default [16, 16, 12, 8, 4, 4]
        List of hidden dimensions for each linear block
        (feature_map, conv1, pool1, conv2, pool2, conv3).
    activation : str, default "tanh"
        Activation function name.
    dropout : float, default 0.0
        Dropout probability applied after every linear block.
    residual : bool, default False
        If True, add skip connections between consecutive blocks.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: Sequence[int] | None = None,
        activation: str = "tanh",
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        if len(hidden_dims)!= 6:
            raise ValueError("hidden_dims must contain six elements")

        self.hidden_dims = hidden_dims
        act = _get_activation(activation)

        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), act, nn.Dropout(dropout)
        )
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]), act, nn.Dropout(dropout)
        )
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]), act, nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[3]), act, nn.Dropout(dropout)
        )
        self.pool2 = nn.Sequential(
            nn.Linear(hidden_dims[3], hidden_dims[4]), act, nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Linear(hidden_dims[4], hidden_dims[5]), act, nn.Dropout(dropout)
        )
        self.head = nn.Linear(hidden_dims[5], 1)

        self.residual = residual
        if residual:
            self._add_residual_blocks()

    def _add_residual_blocks(self) -> None:
        """Wrap each block to add a residual connection from its input."""
        hd = self.hidden_dims

        def residual_wrapper(block: nn.Module, in_dim: int, out_dim: int) -> nn.Module:
            class ResidualBlock(nn.Module):
                def __init__(self, block, in_dim, out_dim):
                    super().__init__()
                    self.block = block
                    self.res = nn.Linear(in_dim, out_dim)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.block(x) + self.res(x)

            return ResidualBlock(block, in_dim, out_dim)

        self.feature_map = residual_wrapper(self.feature_map, 8, hd[0])
        self.conv1 = residual_wrapper(self.conv1, hd[0], hd[1])
        self.pool1 = residual_wrapper(self.pool1, hd[1], hd[2])
        self.conv2 = residual_wrapper(self.conv2, hd[2], hd[3])
        self.pool2 = residual_wrapper(self.pool2, hd[3], hd[4])
        self.conv3 = residual_wrapper(self.conv3, hd[4], hd[5])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
