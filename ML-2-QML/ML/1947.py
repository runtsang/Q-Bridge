import torch
from torch import nn

class QCNNModel(nn.Module):
    """
    Depth‑wise, residual‑enabled convolution‑style network that emulates
    quantum convolution layers.  Drop‑out and batch‑norm improve
    generalisation and training stability.
    """

    def __init__(self,
                 input_dim: int = 8,
                 hidden_dims: list[int] | None = None,
                 dropout: float = 0.2,
                 seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4]
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual connections after every two Linear layers
        out = x
        for i in range(0, len(self.net), 4):
            residual = out
            block = self.net[i:i+4]
            out = block(out)
            out = out + residual
        return torch.sigmoid(out)

def QCNN() -> QCNNModel:
    """
    Factory function that returns a ready‑to‑train :class:`QCNNModel`.
    """
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
