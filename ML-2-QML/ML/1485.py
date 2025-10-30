"""ConvHybrid: Classical depth‑wise separable convolution with adaptive thresholding and gating."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class ConvHybrid(nn.Module):
    """
    Classical implementation of a depth‑wise separable convolution
    with a learnable threshold and a gating parameter that fuses
    a quantum output if provided.  It preserves the API of the
    original Conv class (run(data) -> float) while adding
    richer feature extraction.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 1,
        padding: int = 0,
        init_threshold: float = 0.0,
        init_gate: float = 0.5,
    ) -> None:
        """
        Args:
            kernel_size: Size of the square filter.
            stride: Stride of the convolution.
            padding: Zero‑padding added to both sides.
            init_threshold: Initial value for the adaptive threshold.
            init_gate: Initial value for the classical‑quantum gate.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Depth‑wise convolution: 1 input channel, 1 output channel, groups=1
        self.depthwise = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            bias=False,
        )
        # Point‑wise convolution: 1 input channel, 1 output channel
        self.pointwise = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )

        # Learnable threshold for the sigmoid activation
        self.threshold = nn.Parameter(torch.tensor(init_threshold, dtype=torch.float32))

        # Learnable gating between classical and quantum outputs
        self.gate = nn.Parameter(torch.tensor(init_gate, dtype=torch.float32))

    def forward(self, x: torch.Tensor, q_out: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass through the depth‑wise separable convolution and
        optional fusion with a quantum output.

        Args:
            x: Input tensor of shape (N, C=1, H, W) or (H, W).
            q_out: Optional quantum output scalar (N, 1).

        Returns:
            Tensor of shape (N,) containing the final scalar output.
        """
        # Ensure batch dimension
        if x.dim() == 2:  # (H, W)
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif x.dim() == 3 and x.size(1) == 1:
            pass  # (N, 1, H, W)
        else:
            raise ValueError("Input must be 2D or 3D with one channel.")

        # Depth‑wise separable convolution
        out = self.depthwise(x)
        out = self.pointwise(out)

        # Adaptive sigmoid threshold
        out = torch.sigmoid(out - self.threshold)

        # Global average pooling to produce a scalar per example
        out = F.adaptive_avg_pool2d(out, 1).view(out.size(0))

        if q_out is not None:
            # Ensure q_out has shape (N,)
            q_out = q_out.view(-1)
            # Blend classical and quantum outputs
            blended = self.gate * q_out + (1.0 - self.gate) * out
            return blended
        else:
            return out

    def run(self, data: torch.Tensor | list | tuple) -> float:
        """
        Convenience wrapper to mimic the original Conv.run signature.
        Accepts a 2‑D array and returns a scalar.

        Args:
            data: 2‑D array or list of lists.

        Returns:
            float: Classical output scalar.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        else:
            raise ValueError("Data must be a 2‑D array.")
        with torch.no_grad():
            out = self.forward(tensor)
        return out.item()

__all__ = ["ConvHybrid"]
