"""
FullyConnectedLayer (FCL) – Classical neural network with multi‑layer support.

Features
--------
* Multi‑layer feed‑forward architecture (input → hidden → output).
* Parameter vector interface compatible with the original seed.
* Gradient extraction via PyTorch autograd.
* Simple loss and training utilities for quick experiments.
"""

import torch
from torch import nn
import numpy as np
from typing import Iterable, Tuple

class FCL(nn.Module):
    """
    A fully‑connected neural network that accepts a flat parameter vector.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_hidden : int, optional
        Size of the hidden layer. Defaults to 10.
    n_output : int, optional
        Number of output neurons. Defaults to 1.
    """
    def __init__(self, n_features: int = 1, n_hidden: int = 10, n_output: int = 1) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Layers
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

        # Activation
        self.act = nn.Tanh()

        # Parameter bookkeeping
        self.param_shapes = [
            self.fc1.weight.shape,
            self.fc1.bias.shape,
            self.fc2.weight.shape,
            self.fc2.bias.shape,
        ]

    def _flatten_params(self) -> torch.Tensor:
        """Return a flattened vector of all learnable parameters."""
        return torch.cat([p.view(-1) for p in self.parameters()])

    def _set_params_from_vector(self, theta: Iterable[float]) -> None:
        """Set the network parameters from a flat vector."""
        theta = torch.as_tensor(list(theta), dtype=torch.float32)
        assert theta.numel() == self._flatten_params().numel(), (
            f"Expected {self._flatten_params().numel()} parameters, got {theta.numel()}"
        )
        offset = 0
        for p, shape in zip(self.parameters(), self.param_shapes):
            numel = np.prod(shape)
            p.data = theta[offset : offset + numel].view(shape).clone()
            offset += numel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h = self.act(self.fc1(x))
        y = self.act(self.fc2(h))
        return y

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the network with a supplied parameter vector.

        Parameters
        ----------
        thetas : Iterable[float]
            Flat vector of parameters (weights & biases).

        Returns
        -------
        np.ndarray
            Network output as a 1‑D NumPy array.
        """
        self._set_params_from_vector(thetas)
        # Treat the input as the same vector, reshaped to match n_features.
        # If the vector length differs from n_features, we broadcast the last element.
        input_vec = torch.as_tensor(list(thetas), dtype=torch.float32)
        if input_vec.numel() < self.n_features:
            pad = torch.full((self.n_features - input_vec.numel(),), input_vec[-1])
            input_vec = torch.cat([input_vec, pad])
        elif input_vec.numel() > self.n_features:
            input_vec = input_vec[: self.n_features]
        x = input_vec.view(1, -1)  # batch of 1
        out = self.forward(x)
        return out.detach().numpy().flatten()

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the network output w.r.t. the parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Flat vector of parameters.

        Returns
        -------
        np.ndarray
            Gradient vector of the same length as `thetas`.
        """
        self._set_params_from_vector(thetas)
        input_vec = torch.as_tensor(list(thetas), dtype=torch.float32)
        if input_vec.numel() < self.n_features:
            pad = torch.full((self.n_features - input_vec.numel(),), input_vec[-1])
            input_vec = torch.cat([input_vec, pad])
        elif input_vec.numel() > self.n_features:
            input_vec = input_vec[: self.n_features]
        x = input_vec.view(1, -1)
        x.requires_grad = False
        out = self.forward(x)
        out.backward(torch.ones_like(out))
        grad = self._flatten_params().grad
        return grad.detach().numpy()

    # Small training helper for quick experiments
    def train_one_step(
        self,
        thetas: Iterable[float],
        target: float,
        lr: float = 1e-2,
    ) -> Tuple[float, np.ndarray]:
        """
        Perform a single gradient‑descent step on a scalar loss.

        Parameters
        ----------
        thetas : Iterable[float]
            Current parameter vector.
        target : float
            Desired output.
        lr : float, optional
            Learning rate.

        Returns
        -------
        loss : float
            Scalar loss value.
        updated_thetas : np.ndarray
            Updated parameter vector.
        """
        self._set_params_from_vector(thetas)
        input_vec = torch.as_tensor(list(thetas), dtype=torch.float32)
        if input_vec.numel() < self.n_features:
            pad = torch.full((self.n_features - input_vec.numel(),), input_vec[-1])
            input_vec = torch.cat([input_vec, pad])
        elif input_vec.numel() > self.n_features:
            input_vec = input_vec[: self.n_features]
        x = input_vec.view(1, -1)
        y_pred = self.forward(x)
        loss = nn.functional.mse_loss(y_pred, torch.tensor([[target]], dtype=torch.float32))
        loss.backward()
        with torch.no_grad():
            for p in self.parameters():
                p -= lr * p.grad
        updated = self._flatten_params().detach().numpy()
        return loss.item(), updated

__all__ = ["FCL"]
