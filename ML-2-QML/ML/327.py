"""Enhanced classical regressor with configurable depth, dropout, and batch‑normalization."""

import torch
from torch import nn


class EstimatorQNN(nn.Module):
    """
    A flexible feed‑forward regression network.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input features.
    hidden_dims : list[int] | None, default None
        Sequence of hidden layer sizes. If None, defaults to [8, 4].
    output_dim : int, default 1
        Dimensionality of the output.
    dropout : float, default 0.0
        Drop‑out probability applied after each hidden layer.
    use_batchnorm : bool, default False
        Whether to insert a BatchNorm1d layer after each hidden layer.

    Notes
    -----
    The network is constructed as a `nn.Sequential` pipeline:
        Linear → Tanh → (BatchNorm1d) → (Dropout) → Linear → … → Linear
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | None = None,
        output_dim: int = 1,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [8, 4]
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        return self.net(inputs)


__all__ = ["EstimatorQNN"]
