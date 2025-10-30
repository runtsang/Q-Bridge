from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch convolution producing 4-channel feature maps."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).flatten(1)

class KernelEmbedding(nn.Module):
    """RBF‑kernel embedding against a fixed set of prototypes."""
    def __init__(self, gamma: float = 1.0, prototype_dim: int = 64) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("prototypes", torch.randn(prototype_dim, 4 * 14 * 14))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        diff = features.unsqueeze(1) - self.prototypes.unsqueeze(0)
        return torch.exp(-self.gamma * (diff * diff).sum(dim=2))

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation with a learnable shift."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Linear head followed by the hybrid sigmoid."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class QuanvolutionHybrid(nn.Module):
    """Integrated classical quanvolution pipeline."""
    def __init__(self, in_channels: int = 1, gamma: float = 1.0, prototype_dim: int = 64, shift: float = 0.0) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter(in_channels)
        self.kernel = KernelEmbedding(gamma=gamma, prototype_dim=prototype_dim)
        self.hybrid = Hybrid(in_features=prototype_dim, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.filter(x)
        kernel_feats = self.kernel(feats)
        logits = self.hybrid(kernel_feats)
        probs = torch.cat([logits, 1 - logits], dim=-1)
        return torch.log_softmax(probs, dim=-1)

__all__ = ["QuanvolutionHybrid", "QuanvolutionFilter", "KernelEmbedding", "Hybrid", "HybridFunction"]
