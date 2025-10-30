import torch
from torch import nn
import torch.nn.functional as F

class EstimatorQNN(nn.Module):
    """A deeper feed‑forward regression network with residual connections.

    The architecture mirrors the original EstimatorQNN but adds:
    - Two additional hidden layers (16 and 8 units)
    - Batch‑normalisation after each linear layer
    - Dropout to reduce overfitting
    - A simple residual skip‑connection from the input to the final layer
    """
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [8, 16, 8, 4]
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
        # Residual connection: linear map to match output dim
        self.residual = nn.Linear(input_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        out = self.output(h)
        res = self.residual(x)
        return out + res

__all__ = ["EstimatorQNN"]
