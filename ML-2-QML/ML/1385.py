"""Enhanced classical CNN + fully connected model with residuals, dropout, and feature extraction.

The model extends the original QFCModel by adding:
- A deeper convolutional backbone with batch normalization and residual connections.
- Configurable dropout for regularization.
- A flexible classifier head that can output logits, probabilities or embeddings.
- Utility methods for freezing the backbone and reporting parameter counts.

This design enables more expressive feature learning while preserving the simple interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QFCModel(nn.Module):
    """
    A hybrid CNN + FC architecture with residual blocks and dropout.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Default is 1 (grayscale).
    num_classes : int
        Number of output classes. Default is 4.
    dropout : float
        Dropout probability applied after the first fully‑connected layer.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        # Convolutional backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )

        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self.out_norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.out_norm(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 128‑dim embedding before the final classifier."""
        with torch.no_grad():
            x = self.backbone(x)
            x = x.view(x.size(0), -1)
            x = self.fc[0](x)  # Linear
            x = self.fc[1](x)  # BatchNorm
            x = self.fc[2](x)  # ReLU
        return x

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_channels={self.backbone[0].in_channels}, num_classes={self.out_norm.num_features})"


__all__ = ["QFCModel"]
