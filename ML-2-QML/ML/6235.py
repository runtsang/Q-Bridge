import numpy as np
import torch
from torch import nn
from torch.nn.functional import tanh
from typing import Iterable

class FCL(nn.Module):
    """
    Classical fully‑connected layer that mimics a quantum expectation value.

    Parameters
    ----------
    n_features : int
        Length of the input parameter vector.
    bias : bool
        Whether to include a bias term in the linear transformation.
    """

    def __init__(self, n_features: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=bias)
        # Learnable scaling factor to emulate a quantum readout weight.
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns an expectation‑like scalar.

        Parameters
        ----------
        thetas : torch.Tensor
            1‑D tensor of shape (n_features,).

        Returns
        -------
        torch.Tensor
            Tensor of shape (1,) containing the output.
        """
        linear_out = self.linear(thetas)
        return self.scale * tanh(linear_out)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Convenience wrapper that accepts an iterable and returns a NumPy array.
        """
        thetas_tensor = torch.as_tensor(list(thetas), dtype=torch.float32)
        with torch.no_grad():
            out = self.forward(thetas_tensor).cpu()
        return out.numpy()
