import torch
import torch.nn as nn

class ClassicalPatchExtractor(nn.Module):
    """Extracts 2×2 patches from 28×28 images and embeds them with a learnable conv."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        features = self.conv(x)  # (batch, out_channels, 14, 14)
        return features.view(x.size(0), -1)  # flatten to (batch, out_channels*14*14)

class QuanvolutionHybridNet(nn.Module):
    """Purely classical hybrid network that mirrors the quantum‑augmented architecture.
    It extracts patches with a classical conv, flattens them, and produces a binary probability."""
    def __init__(self):
        super().__init__()
        self.patch_extractor = ClassicalPatchExtractor()
        self.fc = nn.Linear(4 * 14 * 14, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_extractor(x)
        logits = self.fc(x)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["QuanvolutionHybridNet"]
