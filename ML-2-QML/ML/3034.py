import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    """Classical hybrid network combining a quanvolution filter with a QCNN-inspired head."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        # Quanvolution filter: 2×2 patches → 4 feature maps
        self.qfilter = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2)
        # QCNN‑inspired fully‑connected stack
        self.qcnn = nn.Sequential(
            nn.Linear(4 * 14 * 14, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, 4), nn.Tanh(),
            nn.Linear(4, num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x).view(x.size(0), -1)
        logits = self.qcnn(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
