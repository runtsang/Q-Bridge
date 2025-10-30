"""ConvEnhanced: Classical convolution with learnable kernels and optional quantum back‑end.

The module exposes a ``ConvEnhanced`` class that behaves like the seed ``Conv`` but
provides three new features:
* **Multi‑channel input / output** – the class now works with arbitrary
  ``(C_in, H, W)`` tensors.
* **Learnable kernel** – the kernel weights are a trainable ``nn.Parameter`` and are updated by a standard optimizer.
* **Hybrid hook** – a ``quantum_back`` callable can be passed that receives
  the intermediate activations and returns a gradient contribution that
  is back‑propagated through the classical filter.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional


class ConvEnhanced(nn.Module):
    """
    A drop‑in replacement for the seed Conv class, with additional
    trainable and quantum‑aware capabilities.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        init_kernel: Optional[torch.Tensor] = None,
        quantum_back: Optional[Callable[[torch.Tensor], float]] = None,
    ) -> None:
        """
        :param kernel_size: The size of the local 2‑D kernel.
        :param threshold: Threshold for binarising the input before the linear layer.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param init_kernel: Optional tensor to initialise the kernel weights.
        :param quantum_back: Optional callable that receives the activation
            tensor and returns a scalar gradient contribution.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.quantum_back = quantum_back

        # Initialise a learnable kernel
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        if init_kernel is not None:
            self.weight = nn.Parameter(init_kernel.clone().detach())
        else:
            self.weight = nn.Parameter(torch.randn(weight_shape) * 0.02)

        # Bias is optional – keep the same behaviour as the seed
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the convolution, binarise the result and optionally
        forward a quantum‑derived gradient.
        """
        # Classical convolution
        conv_out = F.conv2d(
            x, self.weight, bias=self.bias,
            stride=1, padding=self.kernel_size // 2
        )

        # Binarise via sigmoid and threshold
        bin_out = torch.sigmoid(conv_out - self.threshold)

        # If a quantum back‑end is supplied, obtain a scalar and
        # create a backward hook that adds it to the gradient of the
        # convolution output.
        if self.quantum_back is not None:
            # Run the quantum sub‑module on the activations
            # (flattened to the kernel size for each spatial location)
            batch, oc, h, w = bin_out.shape
            flattened = bin_out.permute(0, 2, 3, 1).reshape(-1, self.kernel_size, self.kernel_size)
            quantum_vals = [self.quantum_back(arr.cpu().numpy()) for arr in flattened]
            q_grad = torch.tensor(quantum_vals, dtype=bin_out.dtype,
                                  device=bin_out.device).reshape(batch, h, w, oc).permute(0, 3, 1, 2)

            # Register a backward hook that adds the quantum contribution
            def _backward_hook(grad):
                return grad + q_grad

            bin_out.register_hook(_backward_hook)

        return bin_out

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, threshold={self.threshold}, in_channels={self.in_channels}, out_channels={self.out_channels}"


__all__ = ["ConvEnhanced"]
