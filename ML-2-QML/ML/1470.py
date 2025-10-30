import torch
from torch import nn

class EstimatorQNNExtended(nn.Module):
    """
    Enhanced feed‑forward regression network.

    Parameters
    ----------
    input_dim : int, default 2
        Number of input features.
    hidden_dims : list[int] | tuple[int,...], default (8, 4)
        Sizes of hidden layers.
    output_dim : int, default 1
        Size of the output layer.
    activation : nn.Module, default nn.Tanh()
        Non‑linearity applied after each hidden layer.
    dropout : float, default 0.0
        Dropout probability applied after each hidden layer.
    use_batchnorm : bool, default False
        Whether to insert BatchNorm1d after each hidden layer.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: list[int] | tuple[int,...] = (8, 4),
                 output_dim: int = 1,
                 activation: nn.Module = nn.Tanh(),
                 dropout: float = 0.0,
                 use_batchnorm: bool = False) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, x: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
        """Convenience wrapper that moves the model and data to ``device``."""
        if device is not None:
            self.to(device)
            x = x.to(device)
        with torch.no_grad():
            return self(x)

__all__ = ["EstimatorQNNExtended"]
