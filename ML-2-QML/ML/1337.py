"""Enhanced classical binary classifier with convolutional, transformer, and dense heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridClassifier(nn.Module):
    """Classical CNN+Transformer binary classifier."""
    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        # Transformer encoder over flattened feature maps
        seq_len = 64 * 8 * 8  # depends on input resolution
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=seq_len, nhead=8, dim_feedforward=256),
            num_layers=2,
        )
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(seq_len, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # Flatten and permute to (seq_len, batch, feature)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer(x)
        # Global average over sequence dimension
        x = x.mean(dim=0)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = ["HybridClassifier"]
