"""Extended classical fully‑connected layer with dropout and batched inference."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class FullyConnectedLayer(nn.Module):
    """
    A two‑layer feed‑forward network with optional dropout.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input feature vector.
    hidden_dim : int, default 32
        Size of the hidden layer.
    dropout : float, default 0.0
        Dropout probability applied after the hidden layer (0 disables dropout).

    The network architecture is:
        Linear(n_features → hidden_dim) → ReLU → Dropout → Linear(hidden_dim → 1)
    The output is passed through a tanh non‑linearity.
    """

    def __init__(self, n_features: int = 1, hidden_dim: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Forward pass for a batch of input thetas.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of input values or a 1‑D array; each value is treated as a separate
            sample in the batch.

        Returns
        -------
        np.ndarray
            1‑D array of network outputs, one per input sample.
        """
        # Convert to tensor and reshape to (batch, 1)
        inputs = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs.squeeze().numpy()


__all__ = ["FullyConnectedLayer"]
