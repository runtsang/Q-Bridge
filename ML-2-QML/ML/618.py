"""Hybrid Conv module – classical CNN core with optional quantum feature map.

The module can be used as a drop‑in replacement for the original Conv.
It exposes a learnable threshold that can be optimised with any
PyTorch optimiser.  The filter accepts batched inputs and supports
in_channels/out_channels arguments.

The class is deliberately lightweight so that it can be inserted
into larger models without modification.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional

# Try to import the quantum implementation.  It will be used only
# when ``use_quantum=True`` to keep the classical path free of
# heavy dependencies.
try:
    from conv_qml import Conv as QuantumConv
except Exception:  # pragma: no cover
    QuantumConv = None


class Conv(nn.Module):
    """
    Args:
        kernel_size (int): Size of the convolution kernel.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        threshold (float): Initial threshold for activation.
        use_quantum (bool): If True, forward will delegate to a
            quantum feature‑map implementation.
        learnable_threshold (bool): If True, ``threshold`` is registered
            as a learnable parameter.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        threshold: float = 0.0,
        use_quantum: bool = False,
        learnable_threshold: bool = False,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_quantum = use_quantum

        # Classical convolution core
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, bias=True
        )

        if learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        else:
            self.threshold = torch.tensor(threshold, dtype=torch.float32)

        if self.use_quantum:
            if QuantumConv is None:
                raise ImportError(
                    "QuantumConv implementation could not be imported. "
                    "Make sure pennylane is installed."
                )
            # Instantiate a quantum feature‑map with the same kernel size
            self.quantum = QuantumConv(
                kernel_size=kernel_size, threshold=self.threshold.item()
            )
        else:
            self.quantum = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        If ``use_quantum`` is True, the input is first mapped to a
        quantum circuit, the expectation value is returned and then
        passed through the classical convolution.  Otherwise the
        classical convolution alone is applied.
        """
        if self.use_quantum:
            batch, C, H, W = x.shape
            assert C == self.in_channels, "Channel mismatch"
            assert H % self.kernel_size == 0 and W % self.kernel_size == 0

            # Unfold to patches
            patches = x.unfold(2, self.kernel_size, self.kernel_size).unfold(
                3, self.kernel_size, self.kernel_size
            )  # shape: (batch, C, out_h, out_w, k, k)
            patches = patches.contiguous().view(-1, self.kernel_size, self.kernel_size)

            # Apply quantum feature‑map
            q_values = torch.tensor(
                [self.quantum.run(patch.detach().cpu().numpy()) for patch in patches],
                dtype=torch.float32,
            )
            q_values = q_values.view(batch, C, H // self.kernel_size, W // self.kernel_size)

            # Combine with classical conv
            out = self.conv(q_values)
        else:
            out = self.conv(x)

        # Apply learnable threshold before sigmoid
        out = torch.sigmoid(out - self.threshold)
        return out

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, threshold={self.threshold.item():.3f}, "
            f"use_quantum={self.use_quantum}"
        )


__all__ = ["Conv"]
