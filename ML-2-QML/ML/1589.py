"""Enhanced classical QCNN implementation with residual connections and dropout."""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Linear layer followed by BatchNorm, ReLU and optional Dropout."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.bn(self.linear(x))))


class QCNNEnhancedModel(nn.Module):
    """Stack of convolutional blocks with residual connections emulating a QCNN."""
    def __init__(
        self,
        feature_dim: int = 8,
        hidden_dims: list[int] = [16, 16, 12, 8, 4, 4],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_map = ConvBlock(feature_dim, hidden_dims[0], dropout)
        self.conv1 = ConvBlock(hidden_dims[0], hidden_dims[1], dropout)
        self.pool1 = ConvBlock(hidden_dims[1], hidden_dims[2], dropout)
        self.conv2 = ConvBlock(hidden_dims[2], hidden_dims[3], dropout)
        self.pool2 = ConvBlock(hidden_dims[3], hidden_dims[4], dropout)
        self.conv3 = ConvBlock(hidden_dims[4], hidden_dims[5], dropout)
        self.head = nn.Linear(hidden_dims[5], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections between successive layers
        out = self.feature_map(x)
        out += self.conv1(out)   # residual after first conv
        out += self.pool1(out)   # residual after first pool
        out += self.conv2(out)   # residual after second conv
        out += self.pool2(out)   # residual after second pool
        out += self.conv3(out)   # residual after third conv
        out = self.head(out)
        return self.sigmoid(out)


def QCNN() -> QCNNEnhancedModel:
    """Factory returning the configured enhanced QCNN model."""
    return QCNNEnhancedModel()


__all__ = ["QCNN", "QCNNEnhancedModel"]
