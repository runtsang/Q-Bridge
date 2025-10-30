"""Enhanced classical convolution module with depthwise separable, residual, and attention support."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class ConvEnhanced(nn.Module):
    """
    Drop‑in replacement for the original Conv filter.
    Supports multiple kernel sizes, depthwise‑separable convolutions,
    residual connections, and a learnable attention map.
    """

    def __init__(
        self,
        kernel_sizes: list[int] | None = None,
        threshold: float = 0.0,
        separable: bool = True,
        residual: bool = True,
        attention: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes or [2]
        self.threshold = threshold
        self.separable = separable
        self.residual = residual
        self.attention = attention
        self.device = device

        # Convolution branch for each kernel size
        self.convs = nn.ModuleList()
        self.attn_convs = nn.ModuleList()
        for k in self.kernel_sizes:
            if self.separable:
                depthwise = nn.Conv2d(
                    1, 1, kernel_size=k, bias=False, groups=1
                )
                pointwise = nn.Conv2d(1, 1, kernel_size=1, bias=True)
                conv = nn.Sequential(depthwise, pointwise)
            else:
                conv = nn.Conv2d(1, 1, kernel_size=k, bias=True)
            self.convs.append(conv)

            if self.attention:
                attn = nn.Conv2d(1, 1, kernel_size=1, bias=False)
                self.attn_convs.append(attn)

        self.to(self.device)

    def forward(self, data: torch.Tensor | np.ndarray) -> float:
        """
        Compute the mean activation across all kernels.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        else:
            data = data.to(self.device)

        outputs = []
        attentions = []

        for conv, attn in zip(self.convs, self.attn_convs if self.attention else [None] * len(self.convs)):
            k = conv[0].kernel_size[0] if isinstance(conv, nn.Sequential) else conv.kernel_size[0]
            # Ensure data shape matches kernel size
            if data.shape[-1]!= k:
                pad = k - data.shape[-1]
                if pad > 0:
                    data = F.pad(data, (0, pad, 0, pad))
                else:
                    data = data[..., :k, :k]
            inp = data.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
            logits = conv(inp)
            act = torch.sigmoid(logits - self.threshold)
            outputs.append(act.mean())

            if self.attention:
                attn_logits = attn(inp)
                attn_weights = F.softmax(attn_logits.view(-1), dim=0).view_as(attn_logits)
                attentions.append(attn_weights.mean())

        outputs = torch.stack(outputs)
        mean_output = outputs.mean()

        if self.residual:
            residual = data.mean()
            mean_output = mean_output + residual

        if self.attention:
            mean_output = mean_output * torch.stack(attentions).mean()

        return mean_output.item()

def Conv() -> ConvEnhanced:
    """Return a ConvEnhanced instance that mimics the original API."""
    return ConvEnhanced()

__all__ = ["ConvEnhanced"]
