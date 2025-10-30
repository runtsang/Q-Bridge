"""Enhanced classical convolutional filter with depthwise separable support and optional L1 regularization."""
from __future__ import annotations

import torch
from torch import nn
from typing import Optional

class Conv(nn.Module):
    """
    Drop‑in replacement for the original Conv filter.
    Supports single‑channel or depthwise separable multi‑channel convolution.
    Returns a scalar feature after applying a sigmoid activation and optional threshold.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        n_channels: int = 1,
        l1_reg: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_channels = n_channels
        self.l1_reg = l1_reg

        if self.n_channels == 1:
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        else:
            self.conv = nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                groups=n_channels,
                bias=True,
            )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Expects data shape (B, C, H, W) or (H, W) for a single patch.
        Returns a scalar tensor.
        """
        if isinstance(data, list):
            data = torch.as_tensor(data, dtype=torch.float32)
        if data.ndim == 2:
            data = data.view(1, 1, self.kernel_size, self.kernel_size)
        elif data.ndim == 3:
            data = data.unsqueeze(1)
        # else assume (B, C, H, W)

        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

    def run(self, data) -> float:
        """
        Run the filter and return a scalar float.
        """
        output = self.forward(torch.as_tensor(data, dtype=torch.float32))
        return output.item()

    def l1_loss(self) -> torch.Tensor:
        """
        Return L1 regularization term if l1_reg is set.
        """
        if self.l1_reg is None:
            return torch.tensor(0.0)
        return self.l1_reg * torch.sum(torch.abs(self.conv.weight))

__all__ = ["Conv"]
