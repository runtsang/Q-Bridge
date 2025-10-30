import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Basic residual block with two convolutional layers."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if in_ch!= out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return F.relu(out)

class QuantumNATEnhanced(nn.Module):
    """Extended CNN‑+‑NN model with residual blocks and multi‑head attention."""
    def __init__(self, in_channels: int = 1, num_classes: int = 4, num_heads: int = 4):
        super().__init__()
        self.block1 = ResidualBlock(in_channels, 8)
        self.block2 = ResidualBlock(8, 16)
        self.pool = nn.MaxPool2d(2)
        self.attention = nn.MultiheadAttention(embed_dim=16*7*7, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(16*7*7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.pool(x)
        bsz = x.shape[0]
        flat = x.view(bsz, -1)
        attn_out, _ = self.attention(flat.unsqueeze(1), flat.unsqueeze(1), flat.unsqueeze(1))
        attn_out = attn_out.squeeze(1)
        out = self.fc(attn_out)
        return self.norm(out)
