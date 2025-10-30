import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Optional

class FCL(nn.Module):
    """
    Enhanced fully‑connected layer with optional bias and activation.
    The layer expects a flat parameter vector `thetas` containing
    the weights (and bias if enabled).  It can be used as a drop‑in
    replacement for the original seed while enabling autograd.
    """

    def __init__(
        self,
        n_features: int = 1,
        bias: bool = True,
        activation: Optional[nn.Module] = nn.Tanh()
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=bias)
        self.activation = activation

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Interpret `thetas` as the weight vector (and bias if present)
        and perform a forward pass on a dummy input of ones.
        """
        thetas = torch.as_tensor(list(thetas), dtype=torch.float32)

        # Compute expected number of parameters
        expected = self.linear.weight.numel()
        if self.linear.bias is not None:
            expected += self.linear.bias.numel()
        if thetas.numel()!= expected:
            raise ValueError(f"Expected {expected} parameters, got {thetas.numel()}")

        idx = 0
        w_shape = self.linear.weight.shape
        w = thetas[idx:idx + w_shape.numel()].reshape(w_shape)
        idx += w_shape.numel()

        if self.linear.bias is not None:
            b = thetas[idx:idx + self.linear.bias.numel()].reshape(self.linear.bias.shape)
        else:
            b = None

        # Temporarily copy parameters
        with torch.no_grad():
            self.linear.weight.copy_(w)
            if self.linear.bias is not None:
                self.linear.bias.copy_(b)

        # Dummy input
        x = torch.ones(1, self.linear.in_features)
        out = self.linear(x)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Legacy run interface that returns a NumPy array.
        """
        return self.forward(thetas).detach().numpy()

__all__ = ["FCL"]
