import torch
import torch.nn as nn
import torch.nn.functional as F

class Quanvolution__gen301(nn.Module):
    """
    Classical quanvolution module.
    Implements a 2×2 convolution with stride 2 followed by a linear classifier.
    The convolution is followed by batch normalization and ReLU to enhance feature
    extraction. The linear head maps the flattened feature map to 10 classes.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 28x28 image -> 14x14 after conv
        self.linear = nn.Linear(out_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 1, 28, 28).

        Returns:
            Log-softmax logits of shape (batch, num_classes).
        """
        features = self.conv(x)
        features = self.bn(features)
        features = self.relu(features)
        features = features.view(features.size(0), -1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["Quanvolution__gen301"]
