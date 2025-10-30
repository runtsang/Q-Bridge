import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

class QuantumNATEnhanced(nn.Module):
    """
    Classical CNN + self‑attention + residual enhancement of the original
    Quantum‑NAT architecture.  The network outputs a 4‑dimensional vector
    suitable for downstream classification or regression tasks.
    """

    def __init__(self) -> None:
        super().__init__()

        # Depthwise‑separable conv block
        self.depthwise = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.pointwise = nn.Conv2d(8, 8, kernel_size=1, bias=False)
        self.bn_dw = nn.BatchNorm2d(8)
        self.bn_pw = nn.BatchNorm2d(8)

        # Standard conv block
        self.conv = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Self‑attention over flattened feature maps
        self.attn_fc1 = nn.Linear(16 * 7 * 7, 128)
        self.attn_fc2 = nn.Linear(128, 16 * 7 * 7)

        # Residual shortcut
        self.res_conv = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)

        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = LayerNorm(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]

        # Depthwise‑separable conv
        x_dw = self.depthwise(x)
        x_dw = self.bn_dw(x_dw)
        x_dw = F.relu(x_dw)
        x_pw = self.pointwise(x_dw)
        x_pw = self.bn_pw(x_pw)
        x_pw = F.relu(x_pw)

        # Residual connection (downsample to match feature map size)
        res = self.res_conv(x)
        res = F.relu(res)

        # Standard conv block
        x = self.conv(x_pw)
        x = x + res  # residual addition

        # Self‑attention
        flat = x.view(bsz, -1)
        attn = self.attn_fc1(flat)
        attn = F.relu(attn)
        attn = torch.sigmoid(self.attn_fc2(attn))
        x = flat * attn

        # Fully connected head
        out = self.fc(x)
        return self.norm(out)


__all__ = ["QuantumNATEnhanced"]
