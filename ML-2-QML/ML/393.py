"""Hybrid CNN with residual blocks and attention for the Quantum‑NAT task.

The architecture extends the original QFCModel by adding a residual block, an
attention module that re‑weights the flattened features, and a final linear
projection.  The module is fully torch‑based and ready for integration into
standard training pipelines."""
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Two‑layer residual block with optional channel expansion."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = nn.Sequential()
        if stride!= 1 or in_ch!= out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)

class QuantumNATEnhanced(nn.Module):
    """CNN backbone with residual blocks, attention, and final projection."""
    def __init__(self, in_channels: int = 1, num_classes: int = 4,
                 embed_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(8, 16),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.attention = nn.Sequential(
            nn.Linear(16 * 7 * 7, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 16 * 7 * 7),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        flat = self.flatten(x)
        attn = self.attention(flat)
        weighted = flat * attn
        out = self.fc(weighted)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
