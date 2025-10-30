import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class FullyConnectedLayerExtended(nn.Module):
    """
    An enhanced fully‑connected layer that mirrors the interface of the seed
    while offering richer behaviour.

    Parameters
    ----------
    n_features : int
        Number of input features (default 1).  The layer contains a linear
        transform, batch‑normalisation, and optional dropout.
    dropout : float, optional
        Drop‑out probability.  If ``None`` dropout is disabled.
    """
    def __init__(self, n_features: int = 1, dropout: float | None = None) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.bn = nn.BatchNorm1d(1)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the expectation of a tanh‑activated linear output over the
        provided parameter set.  The routine is fully vectorised and works
        with both Python iterables and torch tensors.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of parameter values that will be fed through the linear
            layer.

        Returns
        -------
        np.ndarray
            A single‑value array containing the mean activated output.
        """
        # Ensure we have a 2‑D tensor of shape (N, 1)
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)

        # Linear transform → batch‑norm → dropout (if enabled)
        out = self.linear(theta_tensor)
        out = self.bn(out)
        if self.dropout is not None:
            out = self.dropout(out)

        # Activation and expectation
        activated = torch.tanh(out)
        expectation = activated.mean(dim=0)
        return expectation.detach().cpu().numpy()

    # Alias to keep the original public name
    run = forward

__all__ = ["FullyConnectedLayerExtended"]
