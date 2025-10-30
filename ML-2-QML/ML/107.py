"""Enhanced feed‑forward regression network.

This module extends the toy EstimatorQNN by adding depth,
batch normalisation, dropout and L2 regularisation support.
It can be trained with any PyTorch optimiser.

The public API mirrors the original: calling EstimatorQNN()
returns an instance of the model ready for training.
"""

import torch
from torch import nn
from typing import Optional, Sequence, Tuple

class EstimatorQNNModel(nn.Module):
    """A flexible regression network."""
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (16, 8),
        dropout: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)

def EstimatorQNN(
    input_dim: int = 2,
    hidden_dims: Sequence[int] = (16, 8),
    dropout: float = 0.1,
    seed: Optional[int] = None,
) -> EstimatorQNNModel:
    """
    Factory returning a configured regression model.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dims : Sequence[int]
        Sizes of successive hidden layers.
    dropout : float
        Drop‑out probability after each hidden layer.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    EstimatorQNNModel
        Instantiated model ready for training.
    """
    return EstimatorQNNModel(input_dim, hidden_dims, dropout, seed)

__all__ = ["EstimatorQNN"]
