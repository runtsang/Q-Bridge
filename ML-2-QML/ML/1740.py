import torch
from torch import nn
import torch.nn.functional as F

def residual_block(in_dim: int, out_dim: int, drop_prob: float = 0.0):
    """A simple residual block with optional dropout."""
    layers = [
        nn.Linear(in_dim, out_dim),
        nn.ReLU(inplace=True)
    ]
    if drop_prob > 0.0:
        layers.append(nn.Dropout(drop_prob))
    layers.append(nn.Linear(out_dim, out_dim))
    return nn.Sequential(*layers)

class ResidualMLP(nn.Module):
    """Residual MLP for regression with dropout."""
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: list[int] = [8, 8, 4],
                 output_dim: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        blocks = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            blocks.append(residual_block(prev_dim, h_dim, dropout))
            prev_dim = h_dim
        blocks.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def EstimatorQNN() -> ResidualMLP:
    """Return an instance of the residual MLP."""
    return ResidualMLP()

__all__ = ["EstimatorQNN"]
