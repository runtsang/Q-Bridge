"""Enhanced classical quanvolution filter with multi‑scale depth‑wise separable convolutions.

The implementation introduces 2×2, 4×4, and 8×8 patch convolutions using depth‑wise separable layers, a residual shortcut, and a compact linear classifier.  The design is fully torch‑based and can run on CPU or GPU, providing a direct drop‑in replacement for the original QuanvolutionFilter/Classifier pair while improving efficiency and feature richness."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionGen287(nn.Module):
    """Classical quanvolution with multi‑scale depth‑wise separable conv."""
    def __init__(self, in_channels: int = 1, out_classes: int = 10) -> None:
        super().__init__()
        # 2×2 patch conv
        self.dw2 = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2, groups=in_channels)
        self.pw2 = nn.Conv2d(4, 4, kernel_size=1)
        # 4×4 patch conv
        self.dw4 = nn.Conv2d(in_channels, 4, kernel_size=4, stride=4, groups=in_channels)
        self.pw4 = nn.Conv2d(4, 4, kernel_size=1)
        # 8×8 patch conv
        self.dw8 = nn.Conv2d(in_channels, 4, kernel_size=8, stride=8, groups=in_channels)
        self.pw8 = nn.Conv2d(4, 4, kernel_size=1)
        # Residual point‑wise conv to match channel dimension
        self.residual = nn.Conv2d(in_channels, 4, kernel_size=1)
        # Linear head
        self.head = nn.Linear(4 * 3 * 14 * 14, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual
        res = self.residual(x)
        # 2×2 features
        f2 = F.relu(self.pw2(self.dw2(x)))
        # 4×4 features
        f4 = F.relu(self.pw4(self.dw4(x)))
        # 8×8 features
        f8 = F.relu(self.pw8(self.dw8(x)))
        # Concatenate along channel dimension
        out = torch.cat([f2, f4, f8], dim=1) + res
        # Flatten
        flat = out.view(out.size(0), -1)
        logits = self.head(flat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionGen287"]
