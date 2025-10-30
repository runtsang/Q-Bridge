"""Enhanced classical quanvolution filter with optional residual connections and a learnable linear layer.

This module preserves the public API of the original ``QuanvolutionFilter`` and
``QuanvolutionClassifier`` but introduces a residual path and a trainable
linear transformation after the convolution.  The design remains fully
classical and can be used as a drop‑in replacement in existing training
pipelines.

The ``use_residual`` flag controls whether a 1×1 convolution is added to the
input as a shortcut.  The ``learnable_dim`` parameter allows the user to
specify the dimensionality of the post‑convolution linear layer, giving
additional flexibility for feature scaling.

Example
-------
>>> from Quanvolution__gen382 import QuanvolutionFilter, QuanvolutionClassifier
>>> model = QuanvolutionClassifier(use_residual=True, learnable_dim=64)
>>> dummy = torch.randn(8, 1, 28, 28)
>>> out = model(dummy)
>>> out.shape
torch.Size([8, 10])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        use_residual: bool = True,
        learnable_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        if self.use_residual:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        else:
            self.shortcut = None
        self.out_channels = out_channels
        self.learnable_dim = (
            learnable_dim if learnable_dim is not None else self.out_channels * 14 * 14
        )
        self.refine = nn.Linear(self.learnable_dim, self.learnable_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        conv_out = self.conv(x)
        if self.use_residual and self.shortcut is not None:
            shortcut = self.shortcut(x)
            out = conv_out + shortcut
        else:
            out = conv_out
        out = torch.relu(out)
        flat = out.view(x.size(0), -1)
        refined = self.refine(flat)
        return refined

class QuanvolutionClassifier(nn.Module):
    def __init__(
        self,
        use_residual: bool = True,
        learnable_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(
            use_residual=use_residual,
            learnable_dim=learnable_dim,
        )
        self.linear = nn.Linear(
            self.qfilter.learnable_dim, 10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
