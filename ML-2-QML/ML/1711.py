import torch
from torch import nn
import numpy as np
from typing import Iterable

class FullyConnectedLayer(nn.Module):
    """
    Classical fully‑connected layer that mimics a quantum‑style circuit.
    Computes the mean of a tanh activation applied to a linear map.
    Provides convenience methods for batch evaluation and analytic gradients.
    """

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """Return the mean output for a batch of input parameters."""
        return torch.tanh(self.linear(thetas)).mean(dim=0)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Convenience wrapper that accepts a 1‑D list/array of scalars."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return self.forward(values).detach().numpy()

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """Analytic gradient of the output w.r.t. the input parameters."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32,
                                 requires_grad=True).view(-1, 1)
        out = self.forward(values)
        out.backward(torch.ones_like(out))
        return values.grad.detach().numpy()

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Manually set linear weights and bias.
        `weights` shape must be (1, n_features + 1) where the last element is bias.
        """
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(weights[0, :-1], dtype=torch.float32).unsqueeze(0))
            self.linear.bias.copy_(torch.as_tensor(weights[0, -1], dtype=torch.float32))

__all__ = ["FullyConnectedLayer"]
