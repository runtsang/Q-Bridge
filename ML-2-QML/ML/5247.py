import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalConvFilter(nn.Module):
    """Drop‑in replacement for the original `QuanvolutionFilter` using a 2×2 convolution."""
    def __init__(self, in_ch: int = 1, out_ch: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)


class ClassicalQuanvolutionHybrid(nn.Module):
    """Purely classical version of the hybrid architecture."""
    def __init__(self, conv_out_ch: int = 4, num_classes: int = 10):
        super().__init__()
        self.conv = ClassicalConvFilter(out_ch=conv_out_ch)
        self.linear = nn.Linear(conv_out_ch * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["ClassicalConvFilter", "ClassicalQuanvolutionHybrid"]
