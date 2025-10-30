"""Hybrid QCNN model combining classical convolutional layers with a quantum block."""

from __future__ import annotations

import torch
from torch import nn
from typing import List, Optional

class QCNNHybrid(nn.Module):
    """Hybrid QCNN that merges classical convolutional layers with a quantum EstimatorQNN block.

    The architecture follows the original QCNN but injects a quantum variational
    block after the pooling stages.  The quantum block is expected to be
    supplied via the ``qnn`` argument; if omitted, the model remains fully
    classical.  Gradients flow through the quantum block using the
    parameter‑shift rule implemented in :class:`~qml.QNNWrapper`.
    """

    def __init__(
        self,
        feature_dim: int = 8,
        conv_dims: List[int] = [16, 16, 12, 8, 4, 4],
        pool_dims: List[int] = [12, 8, 4],
        qnn: Optional[nn.Module] = None,
        head_dim: int = 1,
    ) -> None:
        super().__init__()
        # Classical feature map
        self.feature_map = nn.Sequential(nn.Linear(feature_dim, conv_dims[0]), nn.Tanh())
        # Convolutional layers
        self.convs = nn.ModuleList()
        in_dim = conv_dims[0]
        for out_dim in conv_dims[1:]:
            self.convs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh()))
            in_dim = out_dim
        # Pooling layers
        self.pools = nn.ModuleList()
        in_dim = conv_dims[-1]
        for out_dim in pool_dims:
            self.pools.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh()))
            in_dim = out_dim
        # Quantum block
        self.qnn = qnn
        # Final head
        self.head = nn.Linear(in_dim, head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for conv in self.convs:
            x = conv(x)
        for pool in self.pools:
            x = pool(x)
        if self.qnn is not None:
            # The quantum block expects a 1‑D batch of feature vectors
            x = x.view(x.size(0), -1)
            q_out = self.qnn(x)
            x = q_out
        return torch.sigmoid(self.head(x))
