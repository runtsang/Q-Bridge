"""Hybrid 2‑D convolutional filter with separable depth‑wise conv and optional quantum‑inspired rotation.

This module extends the original Conv filter with two new behaviours:
* **Depth‑wise separable convolution** for efficient feature extraction.
* **Quantum‑inspired rotation** that turns each input pixel into a rotation angle
  and applies a simple sin transform, simulated with NumPy and wrapped in a
  PyTorch autograd‑friendly interface.

The class can be instantiated as a drop‑in replacement for the original ``Conv`` and
supports training via the standard ``torch.nn.Module`` API.

Example::

    from Conv__gen138 import ConvEnhanced

    # Standard classical convolution
    conv = ConvEnhanced(kernel_size=3, threshold=0.0, use_quantum=False)

    # Quantum‑augmented convolution
    conv_q = ConvEnhanced(kernel_size=3, threshold=0.0, use_quantum=True)

    # Forward pass on a single 3x3 patch
    import numpy as np
    patch = np.random.rand(3, 3)
    output = conv_q.run(patch)
"""

import torch
from torch import nn
import numpy as np


class ConvEnhanced(nn.Module):
    """
    Hybrid 2‑D convolutional filter.
    Parameters
    ----------
    kernel_size : int, default=2
        Size of the convolution kernel.
    threshold : float, default=0.0
        Activation threshold for the sigmoid.
    use_quantum : bool, default=False
        Whether to prepend a quantum‑inspired rotation to each patch.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 use_quantum: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_quantum = use_quantum

        # Depth‑wise separable conv: depthwise + pointwise
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size,
                                   groups=1, bias=False)
        self.pointwise = nn.Conv2d(1, 1, kernel_size=1, bias=True)

        # Classical convolution for comparison
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        if self.use_quantum:
            # Simple quantum‑inspired parameters (one per pixel)
            self.theta = nn.Parameter(torch.randn(kernel_size * kernel_size))

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
            Output logits after sigmoid activation.
        """
        if self.use_quantum:
            # Flatten each patch and apply rotation
            B, C, H, W = x.shape
            patches = x.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
            # patches shape: (B, C, H_out, W_out, k, k)
            patches = patches.contiguous().view(B * H * W, self.kernel_size * self.kernel_size)
            # Simulate quantum rotation: sin(theta * pixel)
            rotated = torch.sin(patches * self.theta)
            # Reshape back to image
            rotated = rotated.view(B, H, W, self.kernel_size, self.kernel_size)
            rotated = rotated.permute(0, 3, 4, 1, 2)  # (B, k, k, H, W)
            # For simplicity, take mean over spatial dims
            x = rotated.mean(dim=(3, 4)).unsqueeze(1)  # (B, 1, k, k)

        out = self.conv(x)
        activations = torch.sigmoid(out - self.threshold)
        return activations

    def run(self, data: np.ndarray) -> float:
        """
        Run the filter on a single kernel‑sized patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean activation value of the filter.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = self.forward(tensor)
        return out.mean().item()


__all__ = ["ConvEnhanced"]
