"""Classical QCNN‑style network with optional shot‑noise simulation.

This module mirrors the quantum QCNN architecture while adding dropout and
a lightweight noise model.  The class is fully compatible with PyTorch
training pipelines and can be instantiated with ``QCNNGen()``.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple


class QCNNGen(nn.Module):
    """
    Classical QCNN‑style model.

    Parameters
    ----------
    input_dim : int, default 8
        Dimensionality of the input feature vector.
    dropout : float, default 0.2
        Dropout probability applied after the final convolutional layer.
    noise_shots : int | None, default None
        If set, adds Gaussian shot noise with variance 1/shots.
    seed : int | None, default None
        Random seed for reproducible noise generation.
    """

    def __init__(
        self,
        input_dim: int = 8,
        dropout: float = 0.2,
        noise_shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)
        self.dropout = nn.Dropout(dropout)
        self.noise_shots = noise_shots
        self.seed = seed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        out = torch.sigmoid(self.head(x))
        if self.noise_shots is not None:
            rng = torch.Generator(device=x.device)
            if self.seed is not None:
                rng.manual_seed(self.seed)
            noise = torch.randn_like(out, generator=rng) * (1 / self.noise_shots) ** 0.5
            out = out + noise
        return out


def QCNNGen() -> QCNNGen:
    """
    Factory returning a freshly initialised :class:`QCNNGen` instance.
    """
    return QCNNGen()


__all__ = ["QCNNGen", "QCNNGen"]
