import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuanvolution(nn.Module):
    """Hybrid classical‑quantum filter and classifier with a residual conv block."""
    def __init__(self) -> None:
        super().__init__()
        # Classical convolutional feature extractor
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Lightweight residual mapping
        self.residual = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
        # Linear head to produce class logits
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract local 2×2 patches via convolution
        features = self.conv(x)
        # Add residual mapping
        features = features + self.residual(features)
        # Flatten and classify
        features = features.view(features.size(0), -1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolution"]
