import numpy as np
import torch
from torch import nn
from typing import Iterable

class FCLayer(nn.Module):
    """
    Classical fully connected layer with a single output neuron.

    The layer can be queried via ``run`` with a list of input angles
    to obtain the mean activation value. It also exposes the standard
    ``forward`` interface so it can seamlessly plug into a PyTorch
    network.
    """

    def __init__(self, n_features: int = 1, activation: callable = torch.tanh) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=True)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the layer on a collection of input angles and return
        the mean activation value.

        Parameters
        ----------
        thetas : iterable of float
            Input values to be processed by the linear layer.

        Returns
        -------
        np.ndarray
            1‑D array containing the mean activation over the provided
            thetas.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        activations = self.activation(self.linear(values))
        return activations.mean(dim=0).detach().numpy()

    def get_params(self) -> np.ndarray:
        """Return the underlying linear weights as a flat NumPy array."""
        return self.linear.weight.detach().numpy().flatten()

    def set_params(self, params: np.ndarray) -> None:
        """Set the underlying linear weights from a flat NumPy array."""
        weights = params.reshape(self.linear.weight.shape)
        self.linear.weight.data = torch.tensor(weights, dtype=torch.float32)

__all__ = ["FCLayer"]
