import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class QuanvolutionGen(nn.Module):
    """Depth‑wise separable quanvolution filter with a trainable linear head."""
    def __init__(self, in_channels=1, out_channels=4, kernel_size=2, stride=2, bias=False):
        super().__init__()
        # Depth‑wise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, groups=in_channels, bias=bias)
        # Point‑wise convolution to mix channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        # Linear classifier
        self.fc = nn.Linear(out_channels * 14 * 14, 10)
        # Initialise weights
        init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')
        init.xavier_normal_(self.fc.weight)
        if self.fc.bias is not None:
            init.zeros_(self.fc.bias)

    def prep(self, x: torch.Tensor) -> torch.Tensor:
        """Standardise input to the range [-1, 1]."""
        return 2.0 * (x.float() / 255.0) - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        dw = self.depthwise(x)
        pw = self.pointwise(dw)
        features = pw.view(pw.size(0), -1)
        logits = self.fc(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionGen"]
