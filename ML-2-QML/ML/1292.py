import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConvBlock(nn.Module):
    """Depthwise‑separable convolution block with batch norm and ReLU."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                                   padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return self.relu(x)

class QuantumNATEnhanced(nn.Module):
    """
    A robust extension of the original Quantum‑NAT CNN‑FC architecture.
    The model consists of a depth‑wise separable convolutional backbone
    followed by a fully‑connected head with dropout and batch‑norm.
    """
    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            DepthwiseSeparableConvBlock(1, 8, stride=1),
            nn.MaxPool2d(2),
            DepthwiseSeparableConvBlock(8, 16, stride=1),
            nn.MaxPool2d(2),
        )
        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
