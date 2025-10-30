import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    """Learnable classical encoder that projects raw images to feature maps."""
    def __init__(self, in_channels: int = 1, out_features: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_features, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_features)
        self.act = nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class QuantumHybridNAT(nn.Module):
    """
    Pure‑classical variant of the hybrid network.
    The final four‑dimensional output is produced by a fully‑connected head
    and normalised with BatchNorm1d.  No quantum primitives are used.
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.encoder = EncoderBlock(in_channels, 16)
        self.features = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = self.encoder(x)
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)

__all__ = ["QuantumHybridNAT"]
