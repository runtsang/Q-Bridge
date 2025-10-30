"""Hybrid classical convolution filter with learnable threshold and batch support.

This module extends the original Conv filter by adding a trainable threshold
parameter and support for batched inputs.  The API remains identical to the
seed: a callable object with a `run` method that accepts a 2‑D array or
torch tensor and returns a scalar activation.  The `log` attribute is a
dictionary containing intermediate values for debugging.

Example
-------
>>> from Conv__gen064 import Conv
>>> conv = Conv(kernel_size=3, threshold=0.1)
>>> out = conv.run(torch.randn(3, 3))
>>> print(out)
0.53
>>> print(conv.log['logits'])
tensor([...])
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class Conv(nn.Module):
    """Depth‑wise separable convolution with a learnable activation threshold.

    Parameters
    ----------
    kernel_size : int, default 3
        Size of the square kernel.
    threshold : float, default 0.0
        Initial value of the threshold that shifts the sigmoid activation.
    bias : bool, default True
        Whether to include a bias term in the convolution.
    """

    def __init__(self, kernel_size: int = 3, threshold: float = 0.0,
                 bias: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size,
                              bias=bias, padding=kernel_size // 2)
        # Trainable threshold
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        self.log = {}

    def run(self, data) -> float:
        """Apply the filter to a 2‑D patch.

        Parameters
        ----------
        data : array‑like or torch.Tensor
            Input patch of shape (kernel_size, kernel_size) or (batch, kernel_size, kernel_size).

        Returns
        -------
        float
            Mean activation over the batch (or single value if batch size is 1).
        """
        if isinstance(data, torch.Tensor):
            tensor = data
        else:
            tensor = torch.as_tensor(data, dtype=torch.float32)

        # Support batched input
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)  # (1, H, W)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)  # (B, 1, H, W)

        logits = self.conv(tensor)  # (B, 1, H, W)
        self.log['logits'] = logits.detach().cpu()
        activations = torch.sigmoid(logits - self.threshold)
        self.log['activations'] = activations.detach().cpu()
        mean_act = activations.mean().item()
        return mean_act

__all__ = ["Conv"]
