"""Resilient classical QCNN with residual connections and regularisation."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple


class QCNNModel(nn.Module):
    """
    A convolution‑inspired fully‑connected network that mimics a quantum convolutional neural network.

    The architecture follows the original seed but adds:
      * Residual connections between every two layers to ease optimisation.
      * Batch‑normalisation after each linear block for stable gradients.
      * Drop‑out (p=0.2) before the classification head to reduce over‑fitting.
      * A two‑stage classifier head (intermediate dense + final sigmoid) to sharpen probability estimates.

    The design keeps the overall parameter count comparable to the seed while improving expressivity.
    """

    def __init__(self, *, input_dim: int = 8, hidden_dims: Tuple[int,...] = (16, 12, 8, 4)) -> None:
        """
        Parameters
        ----------
        input_dim
            Size of the input feature vector.
        hidden_dims
            A sequence of hidden layer sizes.
        """
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Tanh(),
        )

        # Build convolution‑pooling blocks with residuals
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            block = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.Tanh(),
            )
            self.blocks.append(block)

        # Residual connections: each block adds its input to its output
        self.residuals = nn.ModuleList([nn.Identity() for _ in self.blocks])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dims[-1] // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        for block, res in zip(self.blocks, self.residuals):
            y = block(x)
            x = y + res(x)  # residual addition
        return self.classifier(x)


def QCNN() -> QCNNModel:
    """
    Factory returning a freshly initialised :class:`QCNNModel`.

    The returned model is ready for training with any PyTorch optimiser.
    """
    return QCNNModel()


__all__ = ["QCNNModel", "QCNN"]
