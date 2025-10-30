"""QuantumNATEnhanced: classical CNN with residual and attention gating."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """Extended classical model that incorporates residuals, gating and quantum‑encoded features."""
    def __init__(self, num_classes: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        # ---------- CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        # Flattened feature size: 16*7*7 = 784
        self.flatten_size = 16 * 7 * 7
        # Residual projection
        self.residual_proj = nn.Linear(self.flatten_size, hidden_dim)
        # Main path
        self.fc1 = nn.Linear(self.flatten_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Attention gate
        self.gate_fc = nn.Linear(self.flatten_size, hidden_dim)
        # Final classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # Output batch norm
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)          # shape: (bsz, 16, 7, 7)
        flat = feat.view(bsz, -1)        # shape: (bsz, 784)
        # Residual connection
        residual = self.residual_proj(flat)
        # Main path
        out = self.fc1(flat)
        out = out + residual            # residual addition
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        # Attention gating
        gate = torch.sigmoid(self.gate_fc(flat))
        out = out * gate
        # Classification
        out = self.classifier(out)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
