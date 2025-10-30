import numpy as np
import torch
from torch import nn
from typing import Iterable, Optional

class HybridFCConvLayer(nn.Module):
    """
    A hybrid classical layer that combines a 2D convolution with a fully connected
    transformation.  The interface mirrors the original FCL module but accepts
    an optional data argument for the convolutional part.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, n_features: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_features = n_features
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float], data: Optional[np.ndarray] = None) -> float:
        """
        Forward pass.

        Parameters
        ----------
        thetas
            Iterable of parameters for the linear part.
        data
            2‑D array of shape (kernel_size, kernel_size) to feed the convolution.
            If omitted, a zero array is used.

        Returns
        -------
        float
            Combined output of the convolutional activation and the linear
            transformation.
        """
        if data is None:
            data = np.zeros((self.kernel_size, self.kernel_size))
        tensor = torch.as_tensor(data, dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        conv_out = activations.mean().item()

        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        linear_out = torch.tanh(self.linear(theta_tensor)).mean().item()
        return conv_out + linear_out

__all__ = ["HybridFCConvLayer"]
