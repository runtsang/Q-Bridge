import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """
    Classical hybrid model that extends the original Quantum‑NAT architecture.
    It uses a small ResNet‑style residual block followed by a multi‑head
    self‑attention layer before projecting to four output features.
    """

    def __init__(self):
        super().__init__()
        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        # Residual block
        self.res = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
        )
        self.res_act = nn.ReLU(inplace=True)
        # Multi‑head self‑attention
        self.attn = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W]
        features = self.conv(x)
        residual = self.res(features)
        features = self.res_act(features + residual)
        # Prepare sequence for attention: [B, L, C]
        B, C, H, W = features.shape
        seq = features.view(B, C, H * W).transpose(1, 2)
        attn_out, _ = self.attn(seq, seq, seq)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        flat = attn_out.view(B, -1)
        out = self.fc(flat)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
