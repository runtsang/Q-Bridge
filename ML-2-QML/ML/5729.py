"""ConvEnhanced: Trainable convolutional filter with learnable threshold."""
from __future__ import annotations

import torch
from torch import nn, Tensor

class ConvEnhanced(nn.Module):
    """
    Drop‑in replacement for the original Conv filter that learns a convolution
    kernel and a threshold via back‑propagation.  The module exposes a
    ``run`` helper that accepts a 2‑D NumPy array or torch Tensor and
    returns the mean activation as a Python float, making it easy to plug
    into existing pipelines.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        # Store threshold as a learnable parameter
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Standard forward pass. ``x`` must have shape ``(batch, 1, H, W)``.
        """
        logits = self.conv(x)
        return torch.sigmoid(logits - self.threshold)

    def run(self, data) -> float:
        """
        Convenience wrapper that accepts a 2‑D NumPy array or torch Tensor,
        runs a single forward pass and returns the mean activation as a
        Python float.
        """
        if isinstance(data, torch.Tensor):
            tensor = data.float()
        else:
            tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
        with torch.no_grad():
            out = self.forward(tensor)
        return out.mean().item()

    def predict(self, data) -> float:
        """
        Alias for :meth:`run`.  Provided for semantic clarity when the filter
        is used as a classifier.
        """
        return self.run(data)

__all__ = ["ConvEnhanced"]
