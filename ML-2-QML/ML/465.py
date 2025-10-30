"""Classical QCNN with residual connections and attention."""

from __future__ import annotations

import torch
from torch import nn

class QCNNAdvancedModel(nn.Module):
    """
    A deeper, more expressive QCNN that builds on the original architecture.

    The module introduces:
    * A residual block that re‑uses the same linear mapping across multiple layers.
    * Self‑attention on the feature map to re‑weight the attention weights
      for each feature channel.
    * An optional dropout layer to regularise the forward pass.
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        # Feature map
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        # Convolutional layers
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Residual mapping
        self.residual = nn.Linear(8, 4)
        # Self‑attention
        self.attention = nn.MultiheadAttention(embed_dim=4, num_heads=2, batch_first=True)
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        # Output head
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # Residual skip
        res = self.residual(inputs)
        x = x + res
        # Self‑attention
        x = x.unsqueeze(1)  # (batch, 1, 4)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(1)
        x = self.dropout(x)
        return torch.sigmoid(self.head(x))

def QCNNAdvanced(dropout: float = 0.0) -> QCNNAdvancedModel:
    """
    Factory returning a configured QCNNAdvancedModel.
    """
    return QCNNAdvancedModel(dropout=dropout)

__all__ = ["QCNNAdvanced", "QCNNAdvancedModel"]
