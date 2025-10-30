"""ConvAdvanced – classical convolutional filter.

This module implements a drop‑in replacement for the original Conv class.
It provides a single 2‑D convolution followed by a sigmoid activation that is
thresholded.  The implementation is fully classical and can be used in
environments that do not have a quantum backend.

Author: gpt‑oss‑20b
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

__all__ = ["ConvAdvanced"]


class ConvAdvanced(nn.Module):
    """A classical 2‑D convolutional filter with optional thresholding.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square filter.
    threshold : float, default 0.0
        Value subtracted from the convolution output before applying the sigmoid.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: np.ndarray) -> float:
        """Run the filter on a 2‑D array.

        Parameters
        ----------
        data : np.ndarray
            Array of shape ``(kernel_size, kernel_size)`` containing pixel
            intensities.  The values are automatically normalised to ``[0, 1]``.

        Returns
        -------
        float
            Mean activation after sigmoid and threshold.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor / (tensor.max() + 1e-8)  # normalise to [0,1]
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()
