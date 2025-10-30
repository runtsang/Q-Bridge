import numpy as np
import torch
from torch import nn
from torch.nn import init

class FullyConnectedLayer(nn.Module):
    """
    Classical fully‑connected layer with flexible activation and training utilities.

    Parameters
    ----------
    n_features : int, default=1
        Number of input features per example.
    activation : str or callable, default='tanh'
        Activation function applied to the linear output. Accepts a string
        ('tanh','relu','sigmoid', 'linear') or a custom callable.
    weight_init : str, default='xavier'
        Weight initialization scheme: 'xavier', 'kaiming', or 'uniform'.
    """

    def __init__(self, n_features: int = 1,
                 activation: str | callable = 'tanh',
                 weight_init: str = 'xavier'):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self._set_activation(activation)
        self._init_weights(weight_init)

    def _set_activation(self, activation):
        if isinstance(activation, str):
            activations = {
                'tanh': torch.tanh,
               'relu': torch.relu,
               'sigmoid': torch.sigmoid,
                'linear': lambda x: x
            }
            if activation not in activations:
                raise ValueError(f"Unsupported activation: {activation}")
            self.activation = activations[activation]
        elif callable(activation):
            self.activation = activation
        else:
            raise TypeError("activation must be a str or callable")

    def _init_weights(self, scheme):
        if scheme == 'xavier':
            init.xavier_uniform_(self.linear.weight)
        elif scheme == 'kaiming':
            init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        elif scheme == 'uniform':
            init.uniform_(self.linear.weight, a=-0.1, b=0.1)
        else:
            raise ValueError(f"Unsupported weight init scheme: {scheme}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1).
        """
        return self.activation(self.linear(x))

    def run(self, thetas: np.ndarray | list[float]) -> np.ndarray:
        """
        Mimic the quantum interface: compute the mean activation over a batch
        of theta values supplied as a 1‑D array or list.

        Parameters
        ----------
        thetas : np.ndarray or list
            Iterable of shape (batch_size,) containing the theta values.

        Returns
        -------
        np.ndarray
            Array of shape (1,) containing the mean activation.
        """
        if isinstance(thetas, list):
            thetas = np.array(thetas, dtype=np.float32)
        if thetas.ndim!= 1:
            raise ValueError("thetas must be a 1-D array or list")
        with torch.no_grad():
            theta_tensor = torch.from_numpy(thetas).view(-1, 1)
            out = self.forward(theta_tensor)
            mean_val = out.mean().item()
        return np.array([mean_val])

    def gradient(self, thetas: np.ndarray | list[float]) -> np.ndarray:
        """
        Compute the gradient of the mean activation with respect to the thetas
        using autograd. Useful for hybrid training loops.

        Parameters
        ----------
        thetas : np.ndarray or list
            Iterable of shape (batch_size,) containing the theta values.

        Returns
        -------
        np.ndarray
            Gradient array of shape (batch_size,).
        """
        if isinstance(thetas, list):
            thetas = np.array(thetas, dtype=np.float32)
        theta_tensor = torch.from_numpy(thetas).view(-1, 1).requires_grad_(True)
        out = self.forward(theta_tensor)
        mean_val = out.mean()
        mean_val.backward()
        return theta_tensor.grad.numpy().flatten()
