import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATPlus(nn.Module):
    """
    Extended classical model that augments the original CNN‑FC architecture with:
    * depth‑wise separable convolutions and a residual block.
    * an optional linear head that can be swapped with a quantum head.
    """
    def __init__(self, in_channels=1, out_features=4, use_residual=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.use_residual = use_residual

        # depth‑wise separable block
        self.dw_conv = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, groups=in_channels)
        self.dw_bn = nn.BatchNorm2d(8)
        self.dw_relu = nn.ReLU(inplace=True)
        self.pw_conv = nn.Conv2d(8, 8, kernel_size=1, padding=0)
        self.pw_bn = nn.BatchNorm2d(8)
        self.pw_relu = nn.ReLU(inplace=True)

        # standard conv block (same as seed)
        self.conv_block = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # residual connection from input to conv_block output
        if self.use_residual:
            self.res_conv = nn.Conv2d(in_channels, 16, kernel_size=1, padding=0)

        # fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_features)
        )
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # depth‑wise separable conv
        dw = self.dw_conv(x)
        dw = self.dw_bn(dw)
        dw = self.dw_relu(dw)
        pw = self.pw_conv(dw)
        pw = self.pw_bn(pw)
        pw = self.pw_relu(pw)

        # residual path
        if self.use_residual:
            res = self.res_conv(x)
            out = pw + res
        else:
            out = pw

        out = self.conv_block(out)
        flat = out.view(out.size(0), -1)
        out = self.fc(flat)
        return self.norm(out)

__all__ = ["QuantumNATPlus"]
