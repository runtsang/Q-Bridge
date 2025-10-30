import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNatHybrid(nn.Module):
    """Classical CNN + fully‑connected head with optional residual skip."""
    def __init__(self, out_features: int = 4, use_residual: bool = True) -> None:
        super().__init__()
        self.use_residual = use_residual
        # Deep 3‑block feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_features),
        )
        self.out_norm = nn.BatchNorm1d(out_features)
        if use_residual:
            # Project 4‑dim avg‑pool skip to output dimension
            self.skip_proj = nn.Linear(4, out_features)
        else:
            self.skip_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        if self.use_residual:
            skip = F.avg_pool2d(x, kernel_size=14).view(bsz, -1)  # shape (B,4)
            out = out + self.skip_proj(skip)
        return self.out_norm(out)

__all__ = ["QuantumNatHybrid"]
