"""ConvGen: a learnable convolutional filter with global pooling.

This class extends the original fixed‑size filter by providing a full
learnable kernel, bias, and optional global‑pooling operation.  It can
be inserted into any PyTorch model and trained end‑to‑end with a
standard optimizer.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

__all__ = ["ConvGen"]


class ConvGen(nn.Module):
    """
    Learnable convolutional filter that maps a 2‑D patch to a single
    scalar output.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel.
    threshold : float, default 0.0
        Bias applied before the sigmoid activation.
    global_pool : bool, default True
        If True, the output of the convolution is globally averaged
        before the sigmoid activation.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 global_pool: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.global_pool = global_pool
        self.conv = nn.Conv2d(1,
                              1,
                              kernel_size=kernel_size,
                              bias=True)
        # initialise bias to threshold value
        with torch.no_grad():
            self.conv.bias.fill_(threshold)

    def run(self, data: torch.Tensor | np.ndarray) -> float:
        """
        Forward pass that accepts a 2‑D array and returns a scalar.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        data = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(data)
        if self.global_pool:
            logits = logits.mean([2, 3])  # global average pool
        activations = torch.sigmoid(logits - self.threshold)
        return activations.item()

    def forward(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        return torch.tensor(self.run(data), dtype=torch.float32)
