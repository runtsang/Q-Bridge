import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class FCLLayer(nn.Module):
    """
    A fully‑connected neural layer that accepts a list of parameters (thetas)
    and returns the mean tanh activation. Dropout can be enabled for regularisation.
    """
    def __init__(self, n_features: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for autograd support. ``thetas`` should be a 1‑D tensor.
        """
        x = self.linear(thetas)
        x = F.tanh(x)
        x = self.dropout(x)
        return x.mean()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Public API mirroring the original seed. Accepts an iterable of floats,
        converts to a tensor, runs the forward pass and returns a NumPy array.
        """
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32)
        with torch.no_grad():
            result = self.forward(theta_tensor)
        return result.numpy()

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the mean tanh activation with respect to thetas.
        Useful for hybrid optimisation loops.
        """
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32, requires_grad=True)
        result = self.forward(theta_tensor)
        result.backward()
        return theta_tensor.grad.numpy()

__all__ = ["FCLLayer"]
