"""Enhanced classical convolutional filter with multi‑kernel support and learnable threshold."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvFilter(nn.Module):
    """
    Drop‑in replacement for the seed Conv class.

    Parameters
    ----------
    kernel_sizes : sequence[int]
        Sizes of convolution kernels to apply.  Each kernel produces an
        intermediate activation map that is later merged by averaging.
    use_bn : bool, default False
        Whether to insert a 1‑channel BatchNorm after each convolution.
    init_threshold : float
        Initial value for the learnable threshold parameter.
    bias : bool, default True
        Whether each convolution has a bias term.
    """

    def __init__(
        self,
        kernel_sizes: list[int] | tuple[int,...] = (2,),
        use_bn: bool = False,
        init_threshold: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.use_bn = use_bn
        # Create one Conv2d per kernel size
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=k,
                    padding=k // 2,  # preserve spatial size
                    bias=bias,
                )
                for k in self.kernel_sizes
            ]
        )
        # Learnable threshold shared across kernels
        self.threshold = nn.Parameter(torch.tensor(init_threshold, dtype=torch.float32))
        if self.use_bn:
            self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Computes the mean sigmoid activation over all kernels.

        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            Input of shape (H, W) or (1, 1, H, W).  The module expects a
            single‑channel 2‑D patch.

        Returns
        -------
        torch.Tensor
            Scalar activation value.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        # Ensure shape (1, 1, H, W)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3 and x.size(0) == 1:
            x = x.unsqueeze(1)
        elif x.dim()!= 4 or x.size(1)!= 1:
            raise ValueError("expected input shape (H, W) or (1, 1, H, W)")

        activations = []
        for conv in self.convs:
            out = conv(x)
            if self.use_bn:
                out = self.bn(out)
            # Apply learnable threshold before sigmoid
            out = torch.sigmoid(out - self.threshold)
            activations.append(out)

        # Average over all kernels and spatial dimensions
        mean_act = torch.stack(activations).mean()
        return mean_act

    def run(self, data) -> float:
        """
        Compatibility wrapper that mimics the original seed API.

        Parameters
        ----------
        data : array‑like
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean activation value.
        """
        with torch.no_grad():
            return float(self.forward(torch.as_tensor(data, dtype=torch.float32)))

__all__ = ["ConvFilter"]
