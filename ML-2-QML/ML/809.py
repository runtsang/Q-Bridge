import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """Classical hybrid encoder inspired by Quantum‑NAT.

    Combines a 3‑layer convolutional extractor with residual skip connections
    and a fully‑connected projection to an 8‑dimensional embedding.
    """

    def __init__(self):
        super().__init__()
        # Multi‑scale convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Residual path to match the final feature map size
        self.res_conv = nn.Conv2d(1, 32, kernel_size=1, stride=4, padding=0)
        self.res_bn = nn.BatchNorm2d(32)

        # Fully connected projection to 8‑dimensional embedding
        self.fc = nn.Sequential(
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 8),
        )
        self.out_bn = nn.BatchNorm1d(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, H, W) – assumed 28×28
        feat = self.features(x)
        res = self.res_bn(self.res_conv(x))
        feat = feat + res
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        return self.out_bn(out)

__all__ = ["QuantumNATEnhanced"]
