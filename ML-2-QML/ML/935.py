import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QuanvolutionGen216(nn.Module):
    """Classical baseline: 2×2 convolution followed by a linear classifier."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x):
        features = self.conv(x)
        features = features.view(x.size(0), -1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionGen216"]
