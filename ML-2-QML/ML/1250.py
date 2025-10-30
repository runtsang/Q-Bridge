import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionNet(nn.Module):
    """
    Classical implementation of a quanvolutional network.
    Mimics a quantum kernel with a learnable 2×2 convolution followed by
    batch‑normalisation, dropout, a scaling factor and a linear classifier.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        # Scaling factor to emulate the quantum measurement amplitude
        self.scale = nn.Parameter(torch.ones(1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param x: Tensor of shape (B, C, H, W) with values in [0,1].
        :return: Log‑softmax logits of shape (B, num_classes).
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        # Scale the feature map
        x = self.scale * x
        x = self.flatten(x)
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
