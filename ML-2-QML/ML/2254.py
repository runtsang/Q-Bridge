"""Hybrid classical filter inspired by quanvolution and quantum kernels.

The filter extracts 2×2 patches, embeds them via a random orthogonal matrix
(simulating a quantum feature map), then applies a learnable linear layer.
"""

from __future__ import annotations

import torch
from torch import nn

class HybridQuanConvFilter(nn.Module):
    def __init__(self, patch_size: int = 2, hidden_dim: int = 32, threshold: float = 0.0) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.threshold = threshold
        # Extract non-overlapping patches
        self.conv = nn.Conv2d(1, 1, kernel_size=patch_size, stride=patch_size, bias=False)
        # Random orthogonal matrix as a quantum kernel approximation
        orthogonal = torch.randn(patch_size * patch_size, hidden_dim)
        q, _ = torch.linalg.qr(orthogonal)
        self.register_buffer("orthogonal", q)
        self.linear = nn.Linear(hidden_dim, 4)  # output channels per patch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,H,W)
        patches = self.conv(x).view(x.size(0), -1, self.patch_size * self.patch_size)
        # Embed each patch into higher-dimensional space
        embedded = torch.einsum("bpk,kh->bph", patches, self.orthogonal)
        # Apply thresholding to emulate binary measurement
        embedded = torch.where(embedded > self.threshold, torch.ones_like(embedded), torch.zeros_like(embedded))
        # Linear mapping to produce feature map
        out = self.linear(embedded)
        return out

    def run(self, data: torch.Tensor) -> float:
        """Run the filter on a single 2×2 patch and return mean activation."""
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits = self.forward(tensor)
        activations = torch.sigmoid(logits)
        return activations.mean().item()

class HybridQuanConvClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.filter = HybridQuanConvFilter()
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        logits = self.classifier(features)
        return torch.log_softmax(logits, dim=-1)

def Conv() -> HybridQuanConvFilter:
    """Drop-in replacement for the original Conv function."""
    return HybridQuanConvFilter()

__all__ = ["HybridQuanConvFilter", "HybridQuanConvClassifier", "Conv"]
