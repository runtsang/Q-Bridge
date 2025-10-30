import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Simple residual block with two conv layers and a shortcut."""
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class SelfAttention(nn.Module):
    """Scaled dot‑product self‑attention over flattened feature maps."""
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)  # B, N, C
        qkv = self.qkv(x)  # B, N, 3C
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, -1, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, -1, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, -1, self.heads, C // self.heads).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, C)
        out = self.out_proj(out)
        out = out.transpose(1, 2).view(B, C, H, W)
        return out

class QuantumNATExtended(nn.Module):
    """Hybrid CNN with residuals, self‑attention, and a final FC layer."""
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.residual = ResidualBlock(16)
        self.attention = SelfAttention(16)
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        bsz = x.shape[0]
        feat = self.features(x)
        feat = self.residual(feat)
        feat = self.attention(feat)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)

__all__ = ["QuantumNATExtended"]
