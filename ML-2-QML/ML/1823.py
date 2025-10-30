import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class AttentionBlock(nn.Module):
    """Multi‑head self‑attention module that operates on the flattened feature vector."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch, embed_dim); here seq_len = 1
        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        return x


class QuantumNATEnhanced(nn.Module):
    """CNN + attention + residual architecture inspired by Quantum‑NAT."""

    def __init__(self, trainable_features: bool = True):
        super().__init__()
        self.trainable_features = trainable_features

        # Core convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Residual connection on the feature map
        self.residual_conv = nn.Conv2d(16, 16, kernel_size=1)

        # Attention over the flattened feature vector
        flat_dim = 16 * 7 * 7  # 28 -> 14 -> 7 after two pools
        self.attention = AttentionBlock(embed_dim=flat_dim, num_heads=4, dropout=0.1)

        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )

        self.norm = nn.BatchNorm1d(4)

        # Freeze or unfreeze feature extractor
        self.toggle_features(self.trainable_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        feat_map = self.features(x)
        residual = self.residual_conv(feat_map)
        feat_map = feat_map + residual

        # Flatten and prepare for attention
        flat = feat_map.view(feat_map.size(0), -1)                 # (batch, flat_dim)
        seq = flat.unsqueeze(0)                                   # (1, batch, flat_dim)
        attn_out = self.attention(seq)                            # (1, batch, flat_dim)
        attn_out = attn_out.squeeze(0)                            # (batch, flat_dim)

        out = self.fc(attn_out)
        return self.norm(out)

    def toggle_features(self, flag: bool):
        """Enable or disable gradients for the convolutional backbone."""
        self.trainable_features = flag
        for m in self.features.modules():
            for p in m.parameters():
                p.requires_grad = flag
