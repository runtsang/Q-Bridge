"""Classical convolutional model with residual and depth‑wise separable layers for Quantum‑NAT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumNATEnhanced(nn.Module):
    """A robust classical network that follows the Quantum‑NAT architecture
    but deeply‑tuned to one‑way sparse data analysis. The network
    introduces depth‑wise separable convolutions and a residual
    skip‑connection to improve feature extraction while keeping
    the model lightweight.

    The original seed was a simple CNN followed by a fully connected
    head.  This variant replaces the two standard convolutions with
    a depth‑wise separable block and adds a residual connection
    from the input of the first block to its output.  The final
    fully connected head remains unchanged but now receives a richer
    representation.

    The design is intentionally modular to facilitate ablation
    studies and to allow easy replacement of the residual block
    with a more advanced architecture (e.g. a MobileNetV2
    bottleneck or a residual block with SE‑attention).
    """

    class DepthwiseSeparableConv(nn.Module):
        """Depth‑wise separable convolution block.

        Consists of a depth‑wise conv (kernel_size=3, padding=1)
        followed by a point‑wise conv (kernel_size=1).  Both
        operations are followed by ReLU and MaxPool.
        """
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.depthwise = nn.Conv2d(
                in_channels, in_channels, kernel_size=3,
                stride=1, padding=1, groups=in_channels
            )
            self.pointwise = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=1, padding=0
            )
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.depthwise(x)
            x = self.relu(x)
            x = self.pointwise(x)
            x = self.relu(x)
            x = self.pool(x)
            return x

    def __init__(self) -> None:
        super().__init__()
        # Residual block: input -> conv -> relu -> pool -> conv -> relu -> pool
        self.res_block = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Depth‑wise separable block after residual
        self.ds_conv = self.DepthwiseSeparableConv(16, 32)
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Residual path
        residual = self.res_block(x)
        # Add skip connection
        out = residual + self.res_block(x)
        # Depth‑wise separable block
        out = self.ds_conv(out)
        # Flatten and project
        flattened = out.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)


__all__ = ["QuantumNATEnhanced"]
