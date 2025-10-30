"""Unified Quanvolution module combining classical convolution and a quantum‑inspired kernel.

This module merges the classical ConvFilter from Conv.py and the original
QuanvolutionFilter.  It extracts 2×2 patches, applies a small MLP to emulate
the quantum kernel, and concatenates the resulting features with the
classical conv output.  The classifier head remains a single linear layer
followed by log_softmax.

The design allows easy ablation: set `use_quantum` to False to revert to
purely classical behaviour, or to True for hybrid features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedQuanvolutionFilter(nn.Module):
    """Hybrid filter that combines a 2‑pixel classical convolution with a
    quantum‑inspired MLP surrogate.  The MLP is a lightweight surrogate for
    the expensive quantum kernel and can be replaced by a genuine quantum
    circuit if desired."""
    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 2,
        threshold: float = 0.0,
        use_quantum: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.threshold = threshold
        self.use_quantum = use_quantum

        # Classical convolution that extracts 2×2 patches
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=4,  # 4 output channels per patch
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )

        # Quantum‑inspired MLP (4‑output)
        self.qmlp = nn.Sequential(
            nn.Linear(kernel_size * kernel_size, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (N, 1, 28, 28)

        Returns:
            Tensor of concatenated features of shape (N, 4 * 14 * 14 * 2)
        """
        # Classical conv features
        conv_feat = self.conv(x)          # (N, 4, 14, 14)
        conv_feat = conv_feat.view(x.size(0), -1)  # (N, 784)

        # Quantum‑inspired features
        if self.use_quantum:
            # Extract patches for the MLP
            patches = x.unfold(2, self.kernel_size, self.stride) \
                         .unfold(3, self.kernel_size, self.stride)  # (N, 1, 14, 14, 2, 2)
            patches = patches.contiguous().view(-1, self.kernel_size * self.kernel_size)  # (N*196, 4)
            qm_features = self.qmlp(patches)  # (N*196, 4)
            qm_features = qm_features.view(x.size(0), -1)  # (N, 784)
        else:
            qm_features = torch.zeros_like(conv_feat)

        # Concatenate classical and quantum‑inspired outputs
        return torch.cat([conv_feat, qm_features], dim=1)

class UnifiedQuanvolutionClassifier(nn.Module):
    """Classifier that stacks the hybrid filter with a single linear layer."""
    def __init__(self, use_quantum: bool = True) -> None:
        super().__init__()
        self.filter = UnifiedQuanvolutionFilter(use_quantum=use_quantum)
        self.linear = nn.Linear(4 * 14 * 14 * 2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["UnifiedQuanvolutionFilter", "UnifiedQuanvolutionClassifier"]
