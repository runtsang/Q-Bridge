import torch
from torch import nn
import numpy as np

class LinearClip(nn.Module):
    """Linear layer with optional clipping and Tanh activation."""
    def __init__(self, in_features, out_features, clip=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.clip = clip
        self.activation = nn.Tanh()

    def forward(self, x):
        if self.clip:
            self.linear.weight.data.clamp_(-5.0, 5.0)
            self.linear.bias.data.clamp_(-5.0, 5.0)
        return self.activation(self.linear(x))

class ConvHybridNet(nn.Module):
    """Classical convolutional network with optional quantum filter and regression/classification head."""
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 use_quantum: bool = False,
                 regression: bool = True,
                 classification: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_quantum = use_quantum
        if not use_quantum:
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        else:
            # Quantum filter is not implemented in the classical module.
            self.conv = None
        self.regression = regression
        self.classification = classification
        if regression:
            self.head = nn.Sequential(
                LinearClip(1, 8, clip=False),
                LinearClip(8, 4, clip=True),
                LinearClip(4, 1, clip=True)
            )
        elif classification:
            self.head = nn.Sequential(
                LinearClip(1, 1, clip=False),
                nn.Sigmoid()
            )
        else:
            self.head = nn.Identity()

    def forward(self, data: torch.Tensor):
        """
        Args:
            data: Tensor of shape (batch, 1, kernel_size, kernel_size)
        Returns:
            Tensor of shape (batch, 1) for regression, or (batch, 2) for binary classification.
        """
        if self.use_quantum:
            # Placeholder: use classical conv to mimic quantum behavior
            conv_out = torch.sigmoid(self.conv(data) - self.threshold).mean(dim=[2, 3])
        else:
            conv_out = torch.sigmoid(self.conv(data) - self.threshold).mean(dim=[2, 3])
        conv_out = conv_out.view(-1, 1)
        out = self.head(conv_out)
        if self.classification:
            # Convert to probability pair
            prob = torch.sigmoid(out)
            return torch.cat([prob, 1 - prob], dim=-1)
        return out

__all__ = ["ConvHybridNet"]
