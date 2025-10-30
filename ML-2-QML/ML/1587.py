"""HybridConv: Classical depth‑wise separable conv + quantum attention."""

from __future__ import annotations

import torch
from torch import nn

class HybridConv(nn.Module):
    """
    A hybrid convolutional layer that replaces the original Conv filter.
    It applies a depth‑wise separable convolution followed by a 1×1 point‑wise
    projection.  The output of a quantum attention circuit is used as a
    gating signal to weight the depth‑wise feature map before merging with
    the point‑wise conv.  The class is fully differentiable and can be
    integrated into any PyTorch training pipeline.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 2,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        # Depth‑wise separable conv: groups=in_channels makes each channel
        # convolve independently.
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=True,
        )
        # 1×1 point‑wise conv to mix channels
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )
        # Learnable threshold for quantum attention (optional)
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        # Store kernel size for compatibility with quantum circuit
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor, quantum_weight: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid layer.

        Args:
            x: Input feature map of shape (B, C, H, W).
            quantum_weight: Attention tensor of shape (B, C, 1, 1) produced
                by the quantum circuit.  Values are expected to be in [0,1].

        Returns:
            torch.Tensor: The output feature map of shape (B, out_channels, H, W).
        """
        # Apply depth‑wise conv
        dw = self.depthwise(x)
        # Broadcast quantum weight
        dw = dw * quantum_weight
        # Point‑wise conv to merge channels
        out = self.pointwise(dw)
        return out

    def get_qparams(self) -> dict:
        """
        Return dictionary of parameters that the quantum circuit may need
        (e.g., kernel size, threshold).  This is a convenience for the
        QML side to construct a compatible circuit.
        """
        return {
            "kernel_size": self.kernel_size,
            "threshold": self.threshold.item(),
        }

__all__ = ["HybridConv"]
