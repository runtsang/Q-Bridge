"""Extended classical quanvolution model with residual connections and a multi‑task head.

The original seed implemented a simple 2×2 stride‑2 convolution followed by a linear
classification head.  This extension adds a learnable residual shortcut, batch
normalisation, and a dual‑output head that returns both logits and an embedding
vector.  The API is deliberately compatible with the seed so that downstream
training scripts can be reused with minimal changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Classical 2‑D filter with a residual shortcut.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 4
        Number of output channels.
    kernel_size : int, default 2
        Size of the convolution kernel.
    stride : int, default 2
        Stride of the convolution.
    bias : bool, default False
        Whether to include a bias term.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias
        )
        # Residual 1×1 conv keeps the spatial resolution compatible
        self.residual = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        res = self.residual(x)
        out = out + res
        out = self.bn(out)
        out = F.relu(out)
        return out


class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier with a multi‑task head.

    The head produces:
      * ``logits`` – log‑softmax over *num_classes*.
      * ``embed`` – a dense embedding that can be used for other tasks.

    Parameters
    ----------
    num_classes : int, default 10
        Number of classification classes.
    embed_dim : int, default 128
        Dimension of the latent embedding.
    dropout : float, default 0.5
        Dropout probability.
    """
    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        self.fc = nn.Linear(4 * 14 * 14, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.filter(x)          # (B, 4, 14, 14)
        flat = features.view(features.size(0), -1)  # (B, 784)
        embed = F.relu(self.fc(flat))
        embed = self.dropout(embed)
        logits = self.cls_head(embed)
        return {
            "logits": F.log_softmax(logits, dim=-1),
            "embed": embed,
        }


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
