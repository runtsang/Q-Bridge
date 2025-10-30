from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple

__all__ = ["ConvGen102", "Conv"]

class ConvFilter(nn.Module):
    """Single depth‑wise convolutional filter with kernel size `k` and threshold."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(
            1, 1, kernel_size=kernel_size, bias=True, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        return torch.sigmoid(logits - self.threshold)

class ConvGen102(nn.Module):
    """Depth‑wise separable convolutional generator with batch support."""
    def __init__(
        self,
        depth: int = 4,
        kernel_size: int = 2,
        threshold: float = 0.0,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        for _ in range(depth):
            conv = ConvFilter(kernel_size=kernel_size, threshold=threshold)
            bn = nn.BatchNorm2d(1) if batch_norm else nn.Identity()
            do = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
            self.layers.append(nn.Sequential(conv, bn, do))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def run(self, data) -> float:
        """Compute the mean activation over a batch of data."""
        if isinstance(data, (list, tuple)):
            data = torch.stack([torch.tensor(d, dtype=torch.float32) for d in data])
        else:
            data = torch.tensor(data, dtype=torch.float32)

        # Ensure shape (batch, 1, H, W)
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.ndim == 3 and data.shape[0] == 1:
            data = data.unsqueeze(1)
        elif data.ndim == 3 and data.shape[1] == 1:
            pass
        else:
            raise ValueError("Input data must be (H,W) or (batch,H,W) with 1 channel.")

        data = data.to(self.device)
        out = self.forward(data)
        return out.mean().item()

def Conv() -> ConvGen102:
    """Factory that returns a ConvGen102 instance with default configuration."""
    return ConvGen102()
