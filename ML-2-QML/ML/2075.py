import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class FullyConnectedLayer(nn.Module):
    """
    A flexible fully‑connected layer that can act as a simple linear mapping
    or as a small multi‑layer perceptron.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dim : int | None
        If provided, a hidden linear layer is added.
    output_dim : int
        Size of the output tensor.
    dropout : float
        Dropout probability applied after the hidden layer.
    use_batch_norm : bool
        Whether to insert a BatchNorm1d after the hidden linear.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int | None = None,
                 output_dim: int = 1,
                 dropout: float = 0.0,
                 use_batch_norm: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        if hidden_dim is None:
            self.model = nn.Linear(input_dim, output_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_dim)]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.model = nn.Sequential(*layers)

    def forward(self, thetas: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Forward pass that accepts a 1‑D iterable of real numbers.

        Parameters
        ----------
        thetas : array‑like
            Input values that will be fed through the network.

        Returns
        -------
        torch.Tensor
            The output of the network, averaged over the batch dimension.
        """
        if isinstance(thetas, np.ndarray):
            thetas = torch.from_numpy(thetas).float()
        if thetas.ndim == 1:
            thetas = thetas.unsqueeze(-1)
        output = self.model(thetas)
        return output.mean(dim=0)

    def run(self, thetas: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Convenience wrapper that returns a NumPy array.
        """
        with torch.no_grad():
            return self.forward(thetas).cpu().numpy()

def FCL():
    """Return a default FullyConnectedLayer instance."""
    return FullyConnectedLayer(input_dim=1)
