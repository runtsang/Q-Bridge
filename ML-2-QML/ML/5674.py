"""Classical convolutional filter with learnable scaling and activation."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import sigmoid, tanh


class Conv(nn.Module):
    """Drop‑in replacement for a quanvolution layer.

    The original seed exposed a single convolution followed by a sigmoid
    activation.  This extension adds:
    * a learnable scaling factor applied to the convolution output;
    * a tunable bias that can be conditioned on the input;
    * a choice of activation (sigmoid or tanh) controlled by ``use_tanh``.
    The module can be used inside a CNN or as a standalone filter.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 use_tanh: bool = False,
                 scale_init: float | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_tanh = use_tanh

        # Core convolution – same shape as the seed
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Learnable scaling factor and optional bias
        self.scale = nn.Parameter(torch.tensor(scale_init if scale_init is not None
                                                else 1.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Return a single scalar output.

        The method accepts an ``(N, C, H, W)`` tensor and returns a
        mean‑value of the logits after scaling and activation.
        """
        # Ensure shape and type
        if not isinstance(data, torch.Tensor):
            tensor = torch.as_tensor(data, dtype=torch.float32)
        else:
            tensor = data

        # Convolution with bias
        logits = self.conv(tensor)

        # Apply scaling and bias
        logits = logits * self.scale + self.bias

        # Activation choice
        if self.use_tanh:
            activations = tanh(logits - self.threshold)
        else:
            activations = sigmoid(logits - self.threshold)

        return activations.mean()

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, threshold={self.threshold}, " \
               f"use_tanh={self.use_tanh}, scale={self.scale.item():.3f}"
