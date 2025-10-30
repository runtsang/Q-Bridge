python
"""Enhanced fully connected layer with trainable parameters and gradient support.

This module builds on the original simple linear layer by adding
multiple hidden layers, batch‑normalisation, dropout and utilities
to flatten / reshape parameters.  The API mirrors the seed
`FCL()` factory but returns a concrete class that can be
instantiated directly.
"""

import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence

class FCLayer(nn.Module):
    """Multi‑layer perceptron with optional dropout and batch‑norm.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector.
    hidden_dims : Sequence[int] | None
        Sizes of the hidden layers.  If ``None`` a single linear layer
        is created.
    output_dim : int, default 1
        Size of the output vector.
    dropout : float, default 0.0
        Dropout probability applied after each hidden layer.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | None = None,
        output_dim: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        hidden_dims = hidden_dims or []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        # Final linear layer
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def run(self, thetas: Iterable[float] | np.ndarray) -> np.ndarray:
        """Run the network with a flat list of parameters.

        The input ``thetas`` is interpreted as the network input
        (not the weights).  The method returns the network output as
        a NumPy array.
        """
        inp = torch.tensor(thetas, dtype=torch.float32).view(1, -1)
        with torch.no_grad():
            out = self.network(inp)
        return out.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Parameter handling helpers
    # ------------------------------------------------------------------
    def get_parameter_vector(self) -> np.ndarray:
        """Return all learnable parameters as a 1‑D NumPy array."""
        return np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in self.parameters()]
        )

    def set_parameter_vector(self, flat: np.ndarray) -> None:
        """Set the network parameters from a flat array."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            new_val = flat[offset : offset + numel].reshape(p.shape)
            p.data.copy_(torch.from_numpy(new_val))
            offset += numel

    # ------------------------------------------------------------------
    # Gradient utilities
    # ------------------------------------------------------------------
    def compute_gradients(
        self, thetas: Iterable[float] | np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """Return the gradient of MSE loss w.r.t. the learnable parameters.

        Parameters
        ----------
        thetas : array‑like
            Current input vector.
        target : array‑like
            Desired output (same shape as network output).
        """
        self.zero_grad()
        inp = torch.tensor(thetas, dtype=torch.float32).view(1, -1)
        target_t = torch.tensor(target, dtype=torch.float32).view(1, -1)
        out = self.network(inp)
        loss = nn.MSELoss()(out, target_t)
        loss.backward()
        grads = []
        for p in self.parameters():
            grads.append(p.grad.detach().cpu().numpy().ravel())
        return np.concatenate(grads)

    # ------------------------------------------------------------------
    # Simple one‑step training helper
    # ------------------------------------------------------------------
    def train_one_step(
        self,
        thetas: Iterable[float] | np.ndarray,
        target: np.ndarray,
        lr: float = 1e-3,
    ) -> None:
        """Perform a single gradient‑descent update on the network parameters."""
        grads = self.compute_gradients(thetas, target)
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            g = grads[offset : offset + numel].reshape(p.shape)
            p.data -= lr * torch.from_numpy(g)
            offset += numel

__all__ = ["FCLayer"]
