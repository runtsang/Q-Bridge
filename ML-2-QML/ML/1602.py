"""Enhanced classical sampler network with deeper architecture and regularization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNNModel(nn.Module):
    """
    A configurable, deeper neural network for sampling tasks.
    Features:
        * 3 hidden layers with BatchNorm and ReLU
        * Dropout for regularization
        * Optional residual connections
        * Softmax output for probability distribution
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | None = None,
        dropout_p: float = 0.1,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_p))
            in_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 2)
        self.use_residual = use_residual
        if use_residual:
            self.residual = nn.Linear(input_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        if self.use_residual:
            features = features + self.residual(x)
        out = self.output_layer(features)
        return F.softmax(out, dim=-1)


def SamplerQNN() -> SamplerQNNModel:
    """Factory returning a ready‑to‑train SamplerQNNModel instance."""
    return SamplerQNNModel()


__all__ = ["SamplerQNN"]
