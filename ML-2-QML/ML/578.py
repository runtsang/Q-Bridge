"""Enhanced classical Quantum‑NAT model.

This module implements a convolutional backbone followed by a
self‑attention pooling layer and a trainable readout head.
The attention mechanism allows the network to focus on the most
informative spatial regions before the final classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """Convolutional backbone + attention + readout.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels (default 1).
    num_classes : int, optional
        Number of output features (default 4).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.feature_dim = 16 * 7 * 7  # assuming 28×28 input
        # Self‑attention pooling
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1),
        )
        # Readout head
        self.readout = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)                 # (B, 16, 7, 7)
        flat = feat.view(bsz, -1)               # (B, feature_dim)
        att_weights = self.attention(flat)      # (B, 1)
        weighted = flat * att_weights          # (B, feature_dim)
        out = self.readout(weighted)            # (B, num_classes)
        return self.norm(out)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for inference."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

__all__ = ["QuantumNATEnhanced"]
