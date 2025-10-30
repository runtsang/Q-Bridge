"""Enhanced QCNN with residual connections and optional quantum feature map integration.

The model mirrors the original structure but adds:
- A residual block after each convolution.
- A configurable feature map that can be replaced with a quantum feature extractor.
- Batch normalization and dropout for regularization.
"""

import torch
from torch import nn
from typing import Callable, Optional

class QCNNHybrid(nn.Module):
    """
    Classical QCNN-like architecture with residual connections.
    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector.
    hidden_dims : list[int]
        List of hidden dimensions for successive layers.
    feature_map : Optional[Callable[[torch.Tensor], torch.Tensor]]
        Function to transform raw input before feeding into the network.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: Optional[list[int]] = None,
        feature_map: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_map = feature_map or (lambda x: x)
        hidden_dims = hidden_dims or [16, 12, 8, 4, 4]
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            conv = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            # Residual connection if dimensions match
            res = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
            layers.append(nn.ModuleDict({"conv": conv, "res": res}))
            in_dim = out_dim
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for block in self.layers:
            conv_out = block["conv"](x)
            res_out = block["res"](x)
            x = conv_out + res_out
            x = torch.relu(x)
        return torch.sigmoid(self.head(x))

def QCNNHybridFactory(**kwargs) -> QCNNHybrid:
    """Convenience factory for creating a QCNNHybrid instance."""
    return QCNNHybrid(**kwargs)

__all__ = ["QCNNHybrid", "QCNNHybridFactory"]
