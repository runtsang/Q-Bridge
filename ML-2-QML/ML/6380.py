"""Enhanced classical QCNN model with transformer encoder and residual connections."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class QCNNEnhancedModel(nn.Module):
    """Classical counterpart of the QCNN architecture.

    The network first projects the 8‑dimensional input into an embedding
    space, passes it through a lightweight transformer encoder to capture
    long‑range dependencies, and then applies the original sequence of
    fully‑connected layers that mimic the quantum convolution and pooling
    stages.  A residual connection from the initial embedding to the
    final output preserves low‑frequency information that would otherwise
    be lost during the successive nonlinearities.
    """

    def __init__(
        self,
        embed_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        conv_hidden: int = 16,
    ) -> None:
        super().__init__()
        # Feature map
        self.feature_map = nn.Sequential(
            nn.Linear(8, embed_dim),
            nn.Tanh(),
        )

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=conv_hidden)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classical convolution / pooling emulation
        self.conv1 = nn.Sequential(nn.Linear(embed_dim, conv_hidden), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(conv_hidden, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # Skip connection
        self.skip = nn.Linear(embed_dim, 4)

        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)  # (batch, embed_dim)
        # Transformer expects (seq_len, batch, embed_dim); we use seq_len=1
        x_t = x.unsqueeze(0)
        x_t = self.transformer(x_t).squeeze(0)
        # Residual
        resid = self.skip(x_t)
        # Convolution / pooling sequence
        x = self.conv1(x_t)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # Combine with residual
        x = x + resid
        return torch.sigmoid(self.head(x))


def QCNNEnhanced() -> QCNNEnhancedModel:
    """Factory returning the configured :class:`QCNNEnhancedModel`."""
    return QCNNEnhancedModel()


__all__ = ["QCNNEnhanced", "QCNNEnhancedModel"]
