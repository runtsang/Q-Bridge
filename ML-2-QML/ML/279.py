"""Classical depth‑wise separable convolution with residual shortcut."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ConvEnhanced(nn.Module):
    """
    Drop‑in replacement for the original Conv filter that adds depth‑wise
    separability, learnable bias, and an optional residual connection.
    It can be used as a single layer or integrated into larger CNNs.

    Parameters
    ----------
    kernel_size : int or tuple
        Size of the square kernel. If a tuple is supplied it should be
        (height, width). Default is 3.
    threshold : float, optional
        Threshold for binarising the input before convolution.
        If None, no thresholding is applied.
    use_residual : bool, default=True
        Whether to add a skip connection that adds the raw input to the
        convolution output.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int] = 3,
        threshold: float | None = None,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_residual = use_residual

        # Depth‑wise separable conv: depth‑wise conv followed by point‑wise conv
        self.depthwise = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            groups=1,
            bias=True,
        )
        self.pointwise = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that accepts a 4‑D tensor (N, C, H, W).

        Returns
        -------
        torch.Tensor
            The processed tensor.
        """
        if self.threshold is not None:
            x = (x > self.threshold).float()

        out = self.depthwise(x)
        out = self.pointwise(out)

        if self.use_residual:
            # Pad the input if necessary so that dimensions match
            if x.shape[-2:]!= out.shape[-2:]:
                pad = [
                    (0, out.shape[-1] - x.shape[-1]),
                    (0, out.shape[-2] - x.shape[-2]),
                ]
                x = F.pad(x, pad)
            out = out + x

        return out

    def run(self, data: np.ndarray) -> float:
        """
        Utility method that mimics the original API: run the filter on a
        single 2‑D array and return the mean activation.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean activation value.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = self.forward(tensor)
        return out.mean().item()
