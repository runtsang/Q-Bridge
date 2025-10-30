"""Enhanced classical fully connected layer with batch support and gradient capabilities."""

import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence

class FCL(nn.Module):
    """
    Fully Connected Layer that accepts a sequence of parameters (thetas) and
    returns the mean activation. The module can be trained end‑to‑end
    and exposes utilities for parameter extraction and gradient evaluation.
    """
    def __init__(self, n_features: int = 1, n_outputs: int = 1, activation: str = "tanh", dropout: float = 0.0):
        """
        Parameters
        ----------
        n_features : int
            Number of incoming features.
        n_outputs : int
            Number of output neurons.
        activation : str
            Activation function name supported: "tanh", "relu", "sigmoid".
        dropout : float
            Dropout probability applied after the linear transform.
        """
        super().__init__()
        self.linear = nn.Linear(n_features, n_outputs)
        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _get_activation(self, name: str):
        if name == "tanh":
            return torch.tanh
        if name == "relu":
            return torch.relu
        if name == "sigmoid":
            return torch.sigmoid
        raise ValueError(f"Unsupported activation {name}")

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Forward pass using the supplied theta parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameters that will be used to compute the linear transformation.

        Returns
        -------
        torch.Tensor
            Mean activation over the batch dimension.
        """
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32)
        # Reshape if necessary
        if theta_tensor.ndim == 1:
            theta_tensor = theta_tensor.unsqueeze(0)
        # Apply linear layer
        out = self.linear(theta_tensor)
        out = self.activation(out)
        out = self.dropout(out)
        return out.mean(dim=0)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Convenience wrapper that returns a NumPy array.
        """
        return self.forward(thetas).detach().cpu().numpy()

    def parameters_vector(self) -> np.ndarray:
        """Return all learnable parameters as a flat NumPy array."""
        return np.concatenate([p.detach().cpu().numpy().flatten() for p in self.parameters()])

    def set_parameters_vector(self, vec: Sequence[float]) -> None:
        """Load a flat parameter vector into the module."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(torch.tensor(vec[offset:offset+numel], dtype=p.dtype).view_as(p))
            offset += numel

    def gradient(self, thetas: Iterable[float], loss_fn=None, target=None) -> np.ndarray:
        """
        Compute the gradient of the output w.r.t. the supplied thetas.
        If loss_fn and target are provided, compute gradient of loss instead.
        """
        self.zero_grad()
        out = self.forward(thetas)
        if loss_fn is not None and target is not None:
            loss = loss_fn(out, torch.as_tensor(target, dtype=out.dtype))
            loss.backward()
            grads = [p.grad for p in self.parameters()]
        else:
            out.backward(torch.ones_like(out))
            grads = [p.grad for p in self.parameters()]
        return np.concatenate([g.detach().cpu().numpy().flatten() for g in grads])

__all__ = ["FCL"]
