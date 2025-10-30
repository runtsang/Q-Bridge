import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """Hybrid classical model extending the original Quantum‑NAT CNN.

    Features:
    * Multi‑scale convolutional blocks with residual connections.
    * Batch‑normalisation after each conv layer.
    * Fully‑connected head projecting to four output features.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4) -> None:
        super().__init__()
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Residual block
        self.res_block = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
        )
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        self.out_bn = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        residual = x
        x = self.res_block(x)
        x += residual
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.out_bn(x)

__all__ = ["QuantumNATEnhanced"]
