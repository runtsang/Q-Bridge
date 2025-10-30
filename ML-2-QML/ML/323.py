import numpy as np
import torch
from torch import nn
from typing import Iterable

class FCLayer(nn.Module):
    """
    Extended fully‑connected neural layer with a configurable hidden unit.
    The ``run`` method accepts a flat list of parameters that are reshaped
    into the weights and biases of a two‑layer perceptron.  The output is
    the mean of a tanh activation applied to the final linear unit, mimicking
    the behaviour of the original toy example but with additional capacity.
    """
    def __init__(self, n_features: int = 1, hidden_size: int = 10):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        # allocate dummy layers; weights will be overwritten by ``run``
        self.hidden = nn.Linear(n_features, hidden_size, bias=True)
        self.output = nn.Linear(hidden_size, 1, bias=True)

    def _reshape_params(self, thetas: np.ndarray):
        """Map the flat parameter array to the two linear layers."""
        expected = (
            self.n_features * self.hidden_size  # hidden weights
            + self.hidden_size                # hidden bias
            + self.hidden_size * 1             # output weights
            + 1                                # output bias
        )
        if len(thetas)!= expected:
            raise ValueError(f"Expected {expected} parameters, got {len(thetas)}")
        idx = 0
        # hidden weights
        w_h = thetas[idx:idx + self.n_features * self.hidden_size].reshape(
            self.n_features, self.hidden_size)
        idx += self.n_features * self.hidden_size
        # hidden bias
        b_h = thetas[idx:idx + self.hidden_size]
        idx += self.hidden_size
        # output weights
        w_o = thetas[idx:idx + self.hidden_size * 1].reshape(
            self.hidden_size, 1)
        idx += self.hidden_size * 1
        # output bias
        b_o = thetas[idx]
        # assign
        self.hidden.weight.data = torch.tensor(
            w_h, dtype=torch.float32)
        self.hidden.bias.data = torch.tensor(
            b_h, dtype=torch.float32)
        self.output.weight.data = torch.tensor(
            w_o, dtype=torch.float32)
        self.output.bias.data = torch.tensor(
            [b_o], dtype=torch.float32)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Perform a forward pass with the supplied parameters.
        Parameters
        ----------
        thetas : Iterable[float]
            Flat list of parameters matching the shape expected by
            ``_reshape_params``.
        Returns
        -------
        np.ndarray
            Single‑element array containing the mean activation value.
        """
        thetas_np = np.asarray(list(thetas), dtype=np.float32)
        self._reshape_params(thetas_np)
        # dummy input of ones
        x = torch.ones((1, self.n_features), dtype=torch.float32)
        hidden = torch.tanh(self.hidden(x))
        out = self.output(hidden)
        expectation = torch.tanh(out).mean().item()
        return np.array([expectation], dtype=np.float32)

__all__ = ["FCLayer"]
