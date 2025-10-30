import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybridModel(nn.Module):
    """Classical baseline that mimics the quantum-inspired architecture.
    Combines a 2D convolution for patch extraction, deeper CNN layers,
    and a fully connected head with dropout."""
    def __init__(self) -> None:
        super().__init__()
        # 2x2 patch extraction via stride‑2 convolution
        self.patch_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flatten and head
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(16 * 3 * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_conv(x)          # shape: (bsz, 4, 14, 14)
        features = self.features(patches)     # shape: (bsz, 16, 3, 3)
        flattened = self.flatten(features)    # shape: (bsz, 16*3*3)
        logits = self.head(flattened)         # shape: (bsz, 10)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybridModel"]
