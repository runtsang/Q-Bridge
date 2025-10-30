"""ConvGen: Multi‑scale, depth‑wise separable convolutional filter.

This module replaces the simple 2‑D filter from the seed with a
trainable, multi‑kernel bank that can be dropped into a PyTorch
model.  It keeps the original API – a callable ``Conv()`` that
returns an ``nn.Module`` – but adds:

* 3 kernel sizes (1, 3, 5) for multi‑scale feature extraction.
* Optional depth‑wise separable convolution for efficiency.
* Batch‑normalisation after each convolution for stable training.
* A tiny linear head that maps the aggregated feature map to a
  single scalar, matching the seed’s return type.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["Conv"]


def Conv():
    """Return a callable object that emulates the quantum filter with PyTorch ops."""
    class ConvGenFilter(nn.Module):
        def __init__(
            self,
            kernel_sizes: list[int] = [1, 3, 5],
            depthwise: bool = True,
            in_channels: int = 1,
            out_channels: int = 1,
            threshold: float = 0.0,
        ) -> None:
            super().__init__()
            self.kernel_sizes = kernel_sizes
            self.depthwise = depthwise
            self.threshold = threshold

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for k in kernel_sizes:
                conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    padding=k // 2,
                    groups=in_channels if depthwise else 1,
                    bias=True,
                )
                bn = nn.BatchNorm2d(out_channels)
                self.convs.append(conv)
                self.bns.append(bn)

            # Small head to collapse the multi‑scale features into a scalar
            self.head = nn.Linear(out_channels * len(kernel_sizes), 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass for a 4‑D tensor of shape (B, C, H, W).

            Returns a scalar per batch element.
            """
            feats = []
            for conv, bn in zip(self.convs, self.bns):
                out = conv(x)
                out = bn(out)
                out = F.relu(out)
                # Global average pooling
                out = out.mean(dim=[2, 3])
                feats.append(out)
            # Concatenate features from all kernels
            feat = torch.cat(feats, dim=1)
            out = self.head(feat)
            return out.squeeze(-1)

        def run(self, data) -> float:
            """
            Run the filter on a 2‑D array (kernel_size, kernel_size).

            The data is converted to a 1‑channel 4‑D tensor and the
            scalar output of ``forward`` is returned.
            """
            tensor = torch.as_tensor(data, dtype=torch.float32)
            # Add batch and channel dimensions
            tensor = tensor[None, None,...]
            with torch.no_grad():
                out = self.forward(tensor)
            return out.item()

    return ConvGenFilter()
