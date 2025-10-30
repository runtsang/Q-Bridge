from __future__ import annotations

import torch
from torch import nn

class ConvHybrid(nn.Module):
    """
    Classical convolutional filter that can be used as a drop‑in replacement for the original Conv.

    Features:
        * Multi‑channel support (in_channels, out_channels)
        * Optional learnable kernel (learning=True) or frozen kernel
        * Sigmoid activation with a user‑defined threshold
        * Simple API: ConvHybrid().run(data) -> float

    The class mirrors the original API while adding richer capabilities for downstream models.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        threshold: float = 0.0,
        learnable_kernel: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.learnable_kernel = learnable_kernel

        # Classical convolution layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        if learnable_kernel:
            nn.init.xavier_uniform_(self.conv.weight)
        else:
            # Freeze kernel weights – useful for testing or hybrid scenarios
            self.conv.weight.requires_grad = False

    def run(self, data) -> float:
        """
        Run the classical filter on a 2‑D patch.

        Args:
            data: 2‑D array or list of shape (kernel_size, kernel_size).

        Returns:
            float: mean activation after sigmoid thresholding.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

    def kernel(self) -> torch.Tensor:
        """
        Return the current kernel weights.

        Returns:
            torch.Tensor: shape (out_channels, in_channels, kernel_size, kernel_size)
        """
        return self.conv.weight.detach()
