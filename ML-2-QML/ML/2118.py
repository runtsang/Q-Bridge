"""Classical QuanvolutionClassifier with multi‑branch attention and early‑stopping support."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBranchAttentionFilter(nn.Module):
    """Apply a 2×2 convolution, depthwise conv, and self‑attention branch, then concatenate."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        # Conv branch
        self.conv_branch = nn.Conv2d(in_channels, out_channels // 2, kernel_size=kernel_size, stride=stride)
        # Depthwise branch
        self.depthwise_branch = nn.Conv2d(in_channels, out_channels // 4, kernel_size=kernel_size, stride=stride,
                                         groups=in_channels)
        # Attention branch
        self.attn_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels - out_channels // 2 - out_channels // 4, kernel_size=1),
            nn.LayerNorm((out_channels - out_channels // 2 - out_channels // 4, 14, 14)),
            nn.SELU()
        )
        self.proj = nn.Linear(out_channels * 14 * 14, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv_branch(x).view(x.size(0), -1)
        dw_out = self.depthwise_branch(x).view(x.size(0), -1)
        attn_out = self.attn_branch(x).view(x.size(0), -1)
        out = torch.cat([conv_out, dw_out, attn_out], dim=1)
        out = self.proj(out)
        return out


class QuanvolutionClassifier(nn.Module):
    """Hybrid classical filter with attention and a linear head. Supports confidence‑based early stopping."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10, confidence_threshold: float = 0.9):
        super().__init__()
        self.qfilter = MultiBranchAttentionFilter(in_channels=in_channels, out_channels=4)
        self.linear = nn.Linear(4, num_classes)
        self.confidence_threshold = confidence_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions with early‑stopping based on confidence threshold."""
        logits = self.forward(x)
        probs = torch.exp(logits)
        max_conf, preds = probs.max(dim=1)
        mask = max_conf >= self.confidence_threshold
        final_preds = torch.where(mask, preds, torch.full_like(preds, -1))
        return final_preds

    def calibrate(self, temperature: float) -> None:
        """Apply temperature scaling to logits for confidence calibration."""
        self.linear.weight.data /= temperature
        self.linear.bias.data /= temperature


__all__ = ["QuanvolutionClassifier"]
