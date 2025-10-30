import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """
    Classical feed‑forward regressor extended from the original seed.

    The model now contains:
        * A configurable embedding network with two hidden layers,
          ReLU non‑linearities, dropout and optional batch‑norm.
        * An output layer that produces a scalar regression value.
    The architecture is fully trainable with PyTorch autograd and can be
    used as a drop‑in replacement for the original EstimatorQNN.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: tuple[int,...] = (64, 32),
        dropout: float = 0.1,
        batch_norm: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim, bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the regressor.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 1) containing the regression output.
        """
        return self.net(inputs)

__all__ = ["EstimatorQNN"]
