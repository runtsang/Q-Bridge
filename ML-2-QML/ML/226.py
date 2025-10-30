import torch
from torch import nn
import numpy as np

class ConvolutionFilter(nn.Module):
    """
    Classical convolutional filter that mimics a quantum filter.

    The filter applies a learnable 2‑D convolution followed by a
    thresholded sigmoid activation.  The module can be dropped into
    a larger neural network in place of a quantum layer.
    """

    def __init__(self,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 threshold: float = 0.0,
                 bias: bool = True) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the square kernel.
        stride : int
            Stride of the convolution.
        padding : int
            Zero‑padding added to both sides of the input.
        threshold : float
            Value subtracted from logits before sigmoid.
        bias : bool
            Whether to use a bias term.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Activation map after convolution and sigmoid.
        """
        logits = self.conv(x)
        return torch.sigmoid(logits - self.threshold)

    def run(self, data: np.ndarray) -> float:
        """
        Convenience wrapper that accepts a NumPy array and returns
        the mean activation value, matching the original API.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        activations = self.forward(tensor)
        return activations.mean().item()
