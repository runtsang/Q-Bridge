import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter224(nn.Module):
    """Classical quanvolution filter for 224×224 images using 3×3 patches."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, patch_size: int = 3, stride: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, in_channels, 224, 224)
        out = self.conv(x)
        return out.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Classifier that uses the 224×224 quanvolution filter followed by a linear head."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter224(in_channels=in_channels)
        # For 224x224 with patch_size=3, stride=2: output spatial dims = 111x111
        self.linear = nn.Linear(4 * 111 * 111, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter224", "QuanvolutionClassifier"]
