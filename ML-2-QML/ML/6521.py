"""Enhanced classical QCNN model with optional preprocessing and a learnable readout."""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional, Callable


class QCNNEnhancedModel(nn.Module):
    """
    A flexible fully‑connected network that emulates a quantum convolutional
    network.  It supports an optional feature‑mapping layer that can be
    used for dimensionality reduction or data augmentation before the
    convolutional stack.  The readout is a learnable sigmoid‑activated
    output layer, which can be replaced by any other loss‑compatible
    head.
    """

    def __init__(
        self,
        *,
        input_dim: int = 8,
        feature_map: Optional[nn.Module] = None,
        conv_layers: int = 3,
        hidden_dim: int = 16,
        pool_dim: int = 12,
        readout_dim: int = 4,
        out_features: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input vector.
        feature_map : nn.Module, optional
            Optional module that transforms the raw input before the
            convolutional stack.  If ``None`` a default linear + ReLU
            mapping is used.
        conv_layers : int
            Number of convolutional layers in the stack.
        hidden_dim : int
            Width of the hidden layers in the convolutional blocks.
        pool_dim : int
            Width of the pooling layers that reduce dimensionality.
        readout_dim : int
            Width of the intermediate readout layer before the final
            classification head.
        out_features : int
            Number of output logits.
        """
        super().__init__()
        self.feature_map = (
            feature_map
            or nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        )

        # Convolutional stack
        self.conv_stack = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                )
                for _ in range(conv_layers)
            ]
        )

        # Pooling stack
        self.pool_stack = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, pool_dim),
                    nn.Tanh(),
                )
                for _ in range(conv_layers)
            ]
        )

        # Readout
        self.readout = nn.Sequential(
            nn.Linear(pool_dim, readout_dim),
            nn.Tanh(),
            nn.Linear(readout_dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        for conv, pool in zip(self.conv_stack, self.pool_stack):
            x = conv(x)
            x = pool(x)
        return torch.sigmoid(self.readout(x))


def QCNNEnhanced(
    *,
    input_dim: int = 8,
    conv_layers: int = 3,
    hidden_dim: int = 16,
    pool_dim: int = 12,
    readout_dim: int = 4,
    out_features: int = 1,
) -> QCNNEnhancedModel:
    """
    Factory that returns a fully configured :class:`QCNNEnhancedModel`.

    The function mirrors the original ``QCNN`` factory while exposing
    additional knobs for depth and dimensionality.  It is intentionally
    kept lightweight so that the model can be dropped into any
    training script.

    Parameters
    ----------
    input_dim : int, optional
        Dimensionality of the input vector.  Default is 8.
    conv_layers : int, optional
        Number of convolutional layers.  Default is 3.
    hidden_dim : int, optional
        Width of hidden layers.  Default is 16.
    pool_dim : int, optional
        Width of pooling layers. Default is 12.
    readout_dim : int, optional
        Width of the intermediate readout layer. Default is 4.
    out_features : int, optional
        Number of output logits.  Default is 1.

    Returns
    -------
    QCNNEnhancedModel
        A ready‑to‑train PyTorch module.
    """
    return QCNNEnhancedModel(
        input_dim=input_dim,
        conv_layers=conv_layers,
        hidden_dim=hidden_dim,
        pool_dim=pool_dim,
        readout_dim=readout_dim,
        out_features=out_features,
    )


__all__ = ["QCNNEnhancedModel", "QCNNEnhanced"]
