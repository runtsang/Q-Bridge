import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    """
    Classical depth‑wise convolutional network that mimics the structure of the
    original quanvolution filter but replaces the quantum kernel with a
    learnable depth‑wise 2×2 convolution followed by a residual block.
    The network is fully trainable with standard gradient descent.
    """
    def __init__(self):
        super().__init__()
        # Depth‑wise 2×2 convolution (output channels = 4)
        self.depthwise_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(4)
        # Residual block
        self.residual = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4)
        )
        self.relu = nn.ReLU(inplace=True)
        # Classifier head
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of images.
        :param x: Tensor of shape (batch, 1, 28, 28)
        :return: Log‑softmax logits of shape (batch, 10)
        """
        features = self.depthwise_conv(x)
        features = self.bn(features)
        residual = self.residual(features)
        features = features + residual
        features = self.relu(features)
        flat = features.view(x.size(0), -1)
        logits = self.linear(flat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
