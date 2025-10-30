import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    """
    Classical depth‑wise separable convolution followed by a residual head.
    The first layer extracts 2×2 patches via a 1‑channel conv producing 4 channels.
    A two‑layer depth‑wise residual network aggregates the features before the final classifier.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        # Depth‑wise conv that outputs 4 channels per 2×2 patch
        self.depthwise = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2, groups=in_channels)
        # Residual block: depth‑wise convs with skip connection
        self.res_block = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, groups=4),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, groups=4),
            nn.BatchNorm2d(4),
        )
        self.relu = nn.ReLU(inplace=True)
        # Final linear head
        self.fc = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract patches
        features = self.depthwise(x)          # shape: (B, 4, 14, 14)
        # Residual connection
        residual = features
        out = self.res_block(features)
        out = out + residual
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        return F.log_softmax(logits, dim=-1)
