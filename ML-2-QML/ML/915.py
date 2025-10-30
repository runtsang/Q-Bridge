import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Residual convolutional block.

    The block applies two 3×3 convolutions with batch‑norm and ReLU,
    and adds the input (after a 1×1 projection if channel numbers
    differ) to form a residual connection.  The output shape is the
    same as the input, enabling the filter to be inserted in place
    of the original 2×2 kernel.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4, stride: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels!= out_channels or stride!= 1:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  stride=stride, bias=False)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.proj is not None:
            residual = self.proj(residual)
        out += residual
        out = self.relu(out)
        return out

class QuanvolutionClassifier(nn.Module):
    """Classifier that uses QuanvolutionFilter followed by global‑average pooling.

    The network first extracts features with a residual block, then
    applies global‑average pooling to collapse spatial dimensions,
    and finally projects to 10 logits with a linear layer.
    """
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels=1, out_channels=4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
