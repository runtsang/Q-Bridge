"""Hybrid classical convolution filter with depth‑wise separable architecture."""
from __future__ import annotations

import torch
from torch import nn


class ConvFusion(nn.Module):
    """
    Classical depth‑wise separable convolution filter inspired by the original
    Conv filter. It accepts a 2‑D patch of shape (kernel_size, kernel_size) and
    returns a single scalar value.

    The filter uses a single depth‑wise kernel per channel (here only one
    channel) and applies an optional sigmoid activation followed by a
    threshold shift. The structure is deliberately lightweight to keep the
    drop‑in behaviour while offering a more expressive transformation than
    the original 2‑D Conv layer.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        use_sigmoid: bool = True,
        bias: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size
            Size of the square kernel (default: 2).
        threshold
            Value subtracted from the activation before averaging.
        use_sigmoid
            Whether to apply a sigmoid non‑linearity.
        bias
            Whether to include a bias term in the convolution.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_sigmoid = use_sigmoid

        # Depth‑wise separable convolution: one kernel per channel (1 → 1)
        self.depthwise = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            groups=1,
            bias=bias,
            padding=0,
        )

    def run(self, data) -> float:
        """
        Execute the filter on a 2‑D patch.

        Parameters
        ----------
        data
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Scalar output of the filter.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.depthwise(tensor)

        activations = torch.sigmoid(logits) if self.use_sigmoid else logits
        activations = activations - self.threshold
        return activations.mean().item()


__all__ = ["ConvFusion"]
