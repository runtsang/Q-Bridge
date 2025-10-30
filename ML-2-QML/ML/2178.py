"""
Extended convolutional filter with a 400×400 kernel and trainable dense layer.

The design follows the same drop‑in API as the original Conv module:
- ``Conv`` is a factory function that returns a ``nn.Module`` subclass.
- The ``run`` method accepts a 2‑D NumPy array and returns a scalar value.
- The new module uses a 400×400 kernel, a residual (conv‑bn‑relu) block and a dense layer that learns a global feature descriptor.

The implementation can be trained end‑to‑end with PyTorch.
"""

import torch
from torch import nn
import numpy as np

def Conv():
    """Factory that creates a ConvGen400 class instance."""

    class ConvGen400(nn.Module):
        """Convolution‑based filter with 400‑pixel kernel and residual block."""

        def __init__(self, kernel_size: int = 400, threshold: float = 0.0) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold

            # Main convolution that extracts a 400×400 patch
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=False)
            nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")

            # Residual block to refine the patch
            self.res_block = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1),
            )

            # Dense read‑out that learns a global descriptor
            self.dense = nn.Linear(1, 1, bias=False)
            nn.init.xavier_uniform_(self.dense.weight)

        def run(self, data: np.ndarray) -> float:
            """
            Run the filter on a 2‑D array.

            Parameters
            ----------
            data : np.ndarray
                2‑D array with shape (H, W) that contains pixel values in [0, 255].

            Returns
            -------
            float
                Scalar score representing the filtered response.
            """
            # Convert to torch tensor and reshape to a single-channel image
            tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Normalize and extract patch
            patch = self.conv(tensor)
            patch = torch.sigmoid(patch - self.threshold)

            # Residual refinement
            residual = self.res_block(patch)
            patch = patch + residual
            patch = torch.relu(patch)

            # Global average pooling and dense output
            pooled = patch.mean(dim=(2, 3))
            output = self.dense(pooled)
            return output.item()

    return ConvGen400()
