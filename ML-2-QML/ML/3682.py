import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HybridQuanvolutionModel(nn.Module):
    """
    Classical hybrid quanvolution classifier.
    Combines a 2x2 convolutional filter with a random fully-connected projection
    and a linear head, inspired by the classical quanvolution and
    Quantum-NAT fully connected model.
    """
    def __init__(self) -> None:
        super().__init__()
        # 2x2 filter producing 4 feature maps
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        # Random linear projection to 64 features, no bias to emulate random quantum kernel
        self.random_proj = nn.Linear(4 * 14 * 14, 64, bias=False)
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(64, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
        )
        # Final classifier
        self.classifier = nn.Linear(4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)                   # [B,4,14,14]
        flattened = features.view(x.size(0), -1)   # [B,4*14*14]
        projected = self.random_proj(flattened)    # [B,64]
        hidden = self.fc(projected)                # [B,4]
        logits = self.classifier(hidden)           # [B,10]
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionModel"]
