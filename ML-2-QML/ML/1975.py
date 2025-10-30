import numpy as np
import torch
from torch import nn

class FullyConnectedLayer(nn.Module):
    """
    Classical fully‑connected layer that supports multi‑layer perceptrons,
    optional dropout and batch‑normalization.

    Parameters
    ----------
    n_features : int, default 1
        Number of input features.
    hidden_dims : Sequence[int], default (32,)
        Sizes of intermediate hidden layers.
    dropout : float, default 0.0
        Dropout probability applied after each hidden layer.
    batch_norm : bool, default False
        Whether to add a batch‑normalization layer after each hidden layer.
    """

    def __init__(self,
                 n_features: int = 1,
                 hidden_dims: tuple[int,...] = (32,),
                 dropout: float = 0.0,
                 batch_norm: bool = False):
        super().__init__()
        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim
        # Output layer
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, n_features).

        Returns
        -------
        torch.Tensor
            Output of shape (batch, 1).
        """
        return self.network(x)

    def run(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Convenience wrapper for a single forward pass that accepts a NumPy array
        or a torch tensor and returns a NumPy array.

        Parameters
        ----------
        x : np.ndarray | torch.Tensor
            Input data.

        Returns
        -------
        np.ndarray
            Output values as a NumPy array.
        """
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            out = self.forward(x)
        return out.cpu().numpy().squeeze()

__all__ = ["FullyConnectedLayer"]
