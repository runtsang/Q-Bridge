"""Advanced feed-forward regressor with batch normalization and optional residuals."""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional, List


class EstimatorQNN(nn.Module):
    """
    A versatile regression network with batch normalization and optional residual connections.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 1,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, output_dim)
        # Residual only if feature dimension matches input
        self.residual = residual and (in_dim == input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.feature_extractor(x)
        if self.residual:
            out = out + x
        return self.output_layer(out)

    def reset_parameters(self) -> None:
        """Reinitialise all weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)


__all__ = ["EstimatorQNN"]
