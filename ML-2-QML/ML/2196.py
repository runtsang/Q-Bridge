"""Enhanced classical convolution module with depth‑wise separable support and logging.

The original Conv class performed a single 2‑D convolution.  ConvEnhanced expands that idea by exposing
- a depth‑wise separable convolution (useful for resource‑light regimes),
- a flag that enables optional logging of intermediate activations,
- a small utility that aggregates the output over a batch.

It remains drop‑in compatible: ``ConvEnhanced(kernel_size=2, threshold=0.0)`` behaves like the original.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional


class ConvEnhanced(nn.Module):
    """
    Drop‑in replacement for the original Conv filter with additional features.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the square kernel (default 2).
    threshold : float, optional
        Threshold applied before the sigmoid activation (default 0.0).
    depthwise : bool, optional
        If True, use a depth‑wise separable 1×1 convolution (default False).
    log_interval : int | None, optional
        If set, prints the mean activation every ``log_interval`` forward calls.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        depthwise: bool = False,
        log_interval: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.depthwise = depthwise
        self.log_interval = log_interval
        self._step = 0

        # The kernel count is always 1 for a drop‑in replacement
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
            groups=1 if not depthwise else kernel_size * kernel_size,
        )

    def forward(self, data: torch.Tensor) -> float:
        """
        Apply the convolution, sigmoid, and return the mean activation.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape ``(H, W)`` or ``(B, H, W)``.

        Returns
        -------
        float
            Mean sigmoid activation over the output feature map.
        """
        if data.dim() == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif data.dim() == 3:
            data = data.unsqueeze(1)  # (B,1,H,W)
        else:
            raise ValueError("Input tensor must be 2D or 3D.")

        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        mean_val = activations.mean().item()

        if self.log_interval is not None and self._step % self.log_interval == 0:
            print(f"[ConvEnhanced] step {self._step} mean activation: {mean_val:.4f}")

        self._step += 1
        return mean_val

    @staticmethod
    def batch_forward(batch: torch.Tensor, **kwargs) -> list[float]:
        """
        Compute the forward pass for a batch of inputs.

        Parameters
        ----------
        batch : torch.Tensor
            Tensor of shape ``(B, H, W)``.

        Returns
        -------
        list[float]
            List of mean activations for each sample in the batch.
        """
        model = ConvEnhanced(**kwargs)
        return [model(sample) for sample in batch]
