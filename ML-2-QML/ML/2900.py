import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvFilter(nn.Module):
    """
    Classical convolutional filter emulating a quantum quanvolution layer.
    Applies a learnable kernel followed by a sigmoid activation and returns
    the mean activation per image as a scalar feature.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3], keepdim=True)  # (batch, 1, 1, 1)

class HybridNATModel(nn.Module):
    """
    Classical hybrid model that mirrors the structure of the original
    Quantum‑NAT but substitutes the first convolutional block with a
    quantum‑inspired filter.  The filter is implemented as a
    ConvFilter, and its scalar output is concatenated with the flattened
    activations of the last convolutional layer before the final
    fully‑connected head.
    """
    def __init__(self) -> None:
        super().__init__()
        # Classical convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.MaxPool2d(2)

        # Quantum‑inspired filter
        self.filter = ConvFilter(kernel_size=2, threshold=0.0)

        # Feature fusion and head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)

        # Quantum‑inspired filter output (scalar per image)
        filt = self.filter(x).view(bsz, 1)

        combined = torch.cat([flat, filt], dim=1)
        out = self.fc(combined)
        return self.norm(out)

__all__ = ["HybridNATModel"]
