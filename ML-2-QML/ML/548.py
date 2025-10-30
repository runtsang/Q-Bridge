import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """Classical attention‑based model inspired by the Quantum‑NAT paper.

    This version extends the original CNN+FC design by adding a multi‑head
    self‑attention encoder, dropout, and batch‑norm.  The attention block
    operates on the flattened feature map produced by a shallow CNN.
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 4,
                 attn_heads: int = 4,
                 attn_dim: int = 32,
                 dropout_prob: float = 0.3):
        super().__init__()
        # Simple feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flattened feature dimension
        self.feature_dim = 16 * 7 * 7  # assuming 28x28 input

        # Multi‑head self‑attention
        self.attn = nn.MultiheadAttention(embed_dim=self.feature_dim,
                                          num_heads=attn_heads,
                                          batch_first=True)
        self.attn_dropout = nn.Dropout(dropout_prob)
        self.attn_norm = nn.LayerNorm(self.feature_dim)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, num_classes),
        )
        self.bn = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.cnn(x)          # shape: (B, 16, 7, 7)
        B, C, H, W = x.shape
        x = x.view(B, -1)        # shape: (B, feature_dim)

        # Self‑attention expects (B, seq_len, embed_dim). Use seq_len=1.
        x_attn, _ = self.attn(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x_attn = x_attn.squeeze(1)
        x_attn = self.attn_dropout(x_attn)
        x_attn = self.attn_norm(x_attn)

        # Classification
        out = self.fc(x_attn)
        return self.bn(out)
