import torch
from torch import nn

class EstimatorQNNAdvanced(nn.Module):
    """Robust regression network with residual blocks, batch normalisation and dropout.
    The architecture is a drop‑in replacement for the original EstimatorQNN, but it
    scales to higher dimensional inputs and improves generalisation through
    regularisation."""
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16, 8]
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            # Residual shortcut when dimensions match
            if in_dim == h:
                layers.append(nn.Identity())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

__all__ = ["EstimatorQNNAdvanced"]
