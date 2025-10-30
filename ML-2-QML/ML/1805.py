import torch
from torch import nn

class EstimatorQNNModel(nn.Module):
    """Flexible feed‑forward regression network with optional dropout,
    layer‑norm, and residual connections.  The architecture is
    configurable through ``input_dim``, ``hidden_dims`` and
    ``dropout``.  It can be used as a drop‑in replacement for the
    original tiny network while still remaining lightweight."""
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: list[int] | tuple[int,...] = (8, 4),
                 output_dim: int = 1,
                 dropout: float = 0.0,
                 activation: nn.Module = nn.Tanh()):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            block = nn.Sequential(
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                activation,
            )
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            layers.append(block)
            # add a residual only when shapes match
            if prev_dim == dim:
                layers.append(ResidualBlock(block))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ResidualBlock(nn.Module):
    """Adds a residual connection: y = x + f(x)."""
    def __init__(self, f: nn.Module):
        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.f(x)

def EstimatorQNN(**kwargs) -> EstimatorQNNModel:
    """Convenience constructor mirroring the original API."""
    return EstimatorQNNModel(**kwargs)

__all__ = ["EstimatorQNN"]
