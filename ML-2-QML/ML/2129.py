"""Hybrid classical convolutional filter with optional quantum backend support."""
from __future__ import annotations

import torch
from torch import nn
import numpy as np

__all__ = ["HybridConv"]


class HybridConv(nn.Module):
    """
    Classical implementation of the Conv filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolutional kernel.
    threshold : float, default 0.0
        Threshold applied after the convolution.  This value is
        treated as a trainable bias if ``learnable=True``.
    learnable : bool, default False
        If True the threshold becomes a learnable parameter.
    device : str | torch.device, default "cpu"
        Device on which to place the module.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        *,
        learnable: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.learnable = learnable
        self.device = torch.device(device)

        # Use a depth‑wise 1‑channel convolution
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
            stride=1,
            padding=0,
        ).to(self.device)

        if learnable:
            self.threshold = nn.Parameter(
                torch.tensor(threshold, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "threshold", torch.tensor(threshold, dtype=torch.float32)
            )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape (B, 1, H, W) or (1, H, W).

        Returns
        -------
        torch.Tensor
            Scalar output per batch element.
        """
        # Ensure 4‑D shape
        if data.ndim == 3:
            data = data.unsqueeze(0)
        data = data.to(self.device)

        logits = self.conv(data)  # shape (B, 1, H-k+1, W-k+1)
        activations = torch.sigmoid(logits - self.threshold)
        # Global average pooling over spatial dims
        out = activations.mean(dim=(2, 3)).squeeze(1)
        return out

    def run(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Convenience wrapper that accepts NumPy arrays and returns a float tensor.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(data).cpu()
