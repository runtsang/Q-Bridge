import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """
    Hybrid classical network with enhanced feature extraction and attention.

    The architecture extends the original CNN+FC design by:
    1. Adding an optional data‑augmentation branch.
    2. Introducing a lightweight multi‑head self‑attention layer
       that operates on the flattened convolutional features.
    3. Using a two‑head transformer block to capture long‑range
       dependencies while keeping inference cost modest.
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 4,
                 augment: bool = True,
                 heads: int = 2,
                 head_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.augment = augment
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.flatten = nn.Flatten()
        # Multi‑head attention: embed_dim = head_dim
        self.attn = nn.MultiheadAttention(embed_dim=head_dim,
                                          num_heads=heads,
                                          batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        # Fully connected head
        # 32 channels * 7 * 7 = 1568 feature dimension
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.augment:
            # Simple horizontal flip with 50% probability
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[-1])
        feat = self.backbone(x)
        bs = feat.size(0)
        feat_flat = self.flatten(feat)          # (bs, 1568)
        seq_len = feat_flat.size(1) // 32      # 49
        embed = feat_flat.view(bs, seq_len, 32) # (bs, 49, 32)
        attn_out, _ = self.attn(embed, embed, embed)
        attn_out = self.attn_drop(attn_out)
        attn_flat = attn_out.reshape(bs, -1)    # (bs, 1568)
        out = self.fc(attn_flat)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
