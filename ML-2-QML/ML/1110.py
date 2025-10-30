"""Enhanced classical QCNN model with residual connections and dropout."""
import torch
from torch import nn
from typing import Optional

class QCNNModel(nn.Module):
    """
    A classical convolution‑inspired neural network that mirrors the structure
    of the quantum QCNN but adds modern regularisation techniques.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    hidden_dims : list[int]
        Sequence of hidden layer sizes for the convolutional blocks.
    dropout : float
        Dropout probability applied after each block.
    use_batchnorm : bool
        Whether to insert a BatchNorm1d layer after each linear.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 12, 8, 4, 4]
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
        )
        blocks = []
        for i in range(len(hidden_dims) - 1):
            layers = [
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.Tanh(),
            ]
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.Dropout(dropout))
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for block in self.blocks:
            residual = x
            x = block(x)
            # residual connection only when dimensions match
            if residual.shape == x.shape:
                x = x + residual
        return torch.sigmoid(self.head(x))

def QCNN() -> QCNNModel:
    """Convenience factory that returns a pre‑configured QCNNModel."""
    return QCNNModel()

__all__ = ["QCNNModel", "QCNN"]
