import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """
    Classical quanvolution filter that applies a 2x2 convolution with
    two independent kernels across the entire image, followed by batch
    normalization and dropout for regularisation.
    """
    def __init__(self, in_channels=1, out_channels=4, kernel_size=2, stride=2, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """
    Hybrid classifier that uses the quanvolution filter followed by a
    fully‑connected head. The filter is trainable and learns a
    representation that mimics the behaviour of a quantum kernel.
    """
    def __init__(self, num_classes=10, dropout=0.2):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # The feature dimension is out_channels * (28/kernel_size)^2
        self.linear = nn.Linear(4 * 14 * 14, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        logits = self.dropout(logits)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
