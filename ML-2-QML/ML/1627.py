import torch
from torch import nn
import numpy as np
from typing import Iterable

class FullyConnectedLayerExtended(nn.Module):
    """
    A multi‑layer fully connected network that mimics the interface of the
    original quantum example.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input.
    n_hidden : int
        Size of the hidden layer.
    dropout : float
        Drop‑out probability after the hidden activation.
    """
    def __init__(self, n_features: int = 1, n_hidden: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return torch.tanh(self.layers(x))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Map a flat list of parameters to the network and evaluate the
        output on a dummy input derived from the same list.

        The interface is deliberately similar to the original seed so that
        downstream experiments can swap the implementation without changing
        the training loop.
        """
        vec = torch.tensor(list(thetas), dtype=torch.float32)
        if vec.numel()!= sum(p.numel() for p in self.parameters()):
            raise ValueError("Parameter vector length mismatch.")
        torch.nn.utils.vector_to_parameters(vec, self.parameters())
        x = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            out = self.forward(x).mean(dim=0)
        return out.detach().numpy()
