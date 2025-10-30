import torch
from torch import nn
import torch.nn.functional as F

class QCNNModel(nn.Module):
    """Enhanced QCNN-inspired neural network with residual connections and dropout.

    The architecture mirrors the original QCNN but adds:
    - Residual skip connections between feature map and convolution blocks.
    - Batch normalization after each linear layer.
    - Dropout for regularization before the final classifier.
    - Configurable depth and hidden sizes.
    """
    def __init__(self,
                 input_dim: int = 8,
                 hidden_dims: list[int] | None = None,
                 dropout: float = 0.2,
                 use_residual: bool = True,
                 device: str | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_residual = use_residual
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Tanh())
            prev_dim = dim
        self.blocks = nn.ModuleList(layers)

        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i in range(0, len(self.blocks), 3):
            block = self.blocks[i:i+3]
            residual = out
            for layer in block:
                out = layer(out)
            if self.use_residual:
                out = out + residual
        out = self.dropout_layer(out)
        out = torch.sigmoid(self.classifier(out))
        return out

def QCNN() -> QCNNModel:
    """Factory returning a default-configured QCNNModel."""
    return QCNNModel()

__all__ = ["QCNNModel", "QCNN"]
