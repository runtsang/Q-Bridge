import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionNet(nn.Module):
    """
    Classical hybrid network inspired by quanvolution.
    It consists of:
        * a 3‑channel 2×2 convolution that produces 32 feature maps
        * a residual block that mixes the feature maps
        * a global average pooling and a linear classifier to 10 classes
    Designed for MNIST‑style (28×28) images but can be adapted to 3‑channel datasets.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        # 2×2 conv stride 2 reduces spatial dimension from 28×28 to 14×14
        self.conv = nn.Conv2d(in_channels, 32, kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Residual block
        self.res_block = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        residual = x
        x = self.res_block(x)
        x += residual
        x = self.relu(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
