"""Hybrid classical convolution module.

Provides a Conv factory that returns a lightweight CNN
followed by a sigmoid activation. The output is a
single float representing the convolution result.
"""

from __future__ import annotations

import torch
from torch import nn

class _HybridConv(nn.Module):
    """Classical CNN used as a drop‑in replacement for quanvolution."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        # 1 filter bank
        self.conv = nn.Conv2d(1, 8, kernel_size=kernel_size, bias=True)
        # Linear layer to collapse features
        self.fc = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolution
        x = self.conv(x)
        # Apply threshold as a sigmoid shift
        x = torch.sigmoid(x - self.threshold)
        # Flatten
        x = torch.flatten(x, start_dim=1)
        # Linear output
        x = self.fc(x)
        # Final sigmoid
        return torch.sigmoid(x)

    def run(self, data) -> float:
        """Run the filter on a 2‑D array."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
        out = self.forward(tensor)
        return out.item()

def Conv():
    """Factory that returns a ConvFilter instance."""
    return _HybridConv()
    
__all__ = ["Conv"]
