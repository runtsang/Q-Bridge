import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionNet(nn.Module):
    """Classical multi‑scale quanvolution network with self‑attention."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        # Multi‑scale convolutions
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=1, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels, 8, kernel_size=5, stride=2, padding=2)
        # Lightweight spatial self‑attention
        self.attention = nn.MultiheadAttention(embed_dim=24, num_heads=4, batch_first=True)
        # Linear head
        self.classifier = nn.Linear(24 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract multi‑scale features
        f1 = self.conv1(x)
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        concat = torch.cat([f1, f3, f5], dim=1)  # (B, 24, 14, 14)
        B, C, H, W = concat.shape
        # Flatten spatial dims for attention
        feat = concat.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        attn_out, _ = self.attention(feat, feat, feat)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        # Classify
        flat = attn_out.view(B, -1)
        logits = self.classifier(flat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
