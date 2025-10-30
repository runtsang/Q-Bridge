"""Classical hybrid model extending Quantum-NAT with transformer encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumNATHybrid(nn.Module):
    """CNN + Transformer encoder + linear head producing 4 features."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        # Transformer encoder operating on flattened feature vector
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32 * 7 * 7, nhead=4, dim_feedforward=128
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Linear head
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)  # (bsz, 32, 7, 7)
        feat = feat.view(bsz, -1)  # (bsz, 32*7*7)
        # Transformer expects (seq_len, batch, d_model); use seq_len=1
        seq = feat.unsqueeze(0)  # (1, bsz, d)
        transformed = self.transformer(seq)  # (1, bsz, d)
        transformed = transformed.squeeze(0)  # (bsz, d)
        out = self.fc(transformed)
        return self.norm(out)


__all__ = ["QuantumNATHybrid"]
