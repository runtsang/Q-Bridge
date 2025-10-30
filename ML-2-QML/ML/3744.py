"""Hybrid classical convolutional network combining convolution filter and QCNN-inspired fully connected layers.

The class inherits from torch.nn.Module and provides a flexible interface that
can be used as a drop‑in replacement for the original Conv filter while
leveraging a lightweight QCNN style architecture for deeper feature extraction.
"""

import torch
from torch import nn
import numpy as np

class Conv(nn.Module):
    """
    Classical hybrid convolutional network.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the 2‑D convolution kernel.
    threshold : float, default 0.0
        Threshold used in the sigmoid activation after the convolution.
    pooling : bool, default True
        If True, the network uses a QCNN‑style sequence of fully connected
        layers that emulate convolution and pooling operations.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, pooling: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.pooling = pooling

        # Simple 2‑D convolution filter
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        if pooling:
            # QCNN‑style linear stack
            self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
            self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
            self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
            self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
            self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
            self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
            self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W) or (1, H, W).
        """
        # Ensure batch dimension
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # Convolution filter
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        activations = activations.view(activations.size(0), -1)  # flatten

        if not self.pooling:
            return activations.mean(dim=1, keepdim=True)

        # QCNN-inspired linear sequence
        x = self.feature_map(activations)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        out = torch.sigmoid(self.head(x))
        return out

    def run(self, data: np.ndarray) -> float:
        """
        Convenience wrapper that accepts a 2‑D NumPy array.

        Parameters
        ----------
        data : np.ndarray
            Input image of shape (H, W).

        Returns
        -------
        float
            Network output as a single scalar.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = self.forward(tensor)
        return output.item()

def ConvFactory(kernel_size: int = 2, threshold: float = 0.0, pooling: bool = True) -> Conv:
    """
    Factory function for creating a :class:`Conv` instance.

    Returns
    -------
    Conv
        Configured hybrid convolutional network.
    """
    return Conv(kernel_size=kernel_size, threshold=threshold, pooling=pooling)

__all__ = ["Conv", "ConvFactory"]
