import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATExtended(nn.Module):
    """
    Classical hybrid model: a CNN backbone, a transformer encoder, and a fully‑connected head.
    Produces a 4‑dimensional output suitable for downstream tasks.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 4, hidden_dim: int = 64):
        super().__init__()
        # Convolutional feature extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # Transformer encoder to capture global interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=hidden_dim, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.backbone(x)                  # (bsz, 32, 1, 1)
        feat = feat.view(bsz, -1)                # (bsz, 32)
        seq = feat.unsqueeze(0)                  # (1, bsz, 32) – seq_len, batch, d_model
        trans_out = self.transformer(seq)        # (1, bsz, 32)
        trans_out = trans_out.squeeze(0)          # (bsz, 32)
        out = self.fc(trans_out)                 # (bsz, num_classes)
        return self.norm(out)

__all__ = ["QuantumNATExtended"]
