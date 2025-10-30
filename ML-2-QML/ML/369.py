"""Enhanced classical feed‑forward regressor with residuals, batch‑norm, and dropout."""
from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass

@dataclass
class EstimatorQNNGen409Config:
    """Hyper‑parameter container."""
    input_dim: int = 2
    hidden_dims: tuple[int,...] = (16, 8)
    output_dim: int = 1
    dropout: float = 0.1
    batchnorm: bool = True
    residual: bool = True

class EstimatorQNNGen409(nn.Module):
    """Classical neural network with optional residuals, batch‑norm and dropout."""
    def __init__(self, cfg: EstimatorQNNGen409Config = EstimatorQNNGen409Config()):
        super().__init__()
        layers = []
        in_dim = cfg.input_dim
        for h_dim in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if cfg.batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.dropout))
            if cfg.residual and h_dim == in_dim:
                layers.append(nn.Identity())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, cfg.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

def EstimatorQNNGen409() -> EstimatorQNNGen409:
    """Convenience factory mirroring the original API."""
    return EstimatorQNNGen409()

__all__ = ["EstimatorQNNGen409", "EstimatorQNNGen409Config"]
