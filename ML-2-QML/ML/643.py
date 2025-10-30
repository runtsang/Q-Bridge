"""Extended classical model: ResNet-like backbone + transformer encoder + MLP."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """A basic residual block with two conv layers."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class QFCModelExtended(nn.Module):
    """Deepened classical model with residual blocks and a lightweight transformer encoder."""
    def __init__(self, num_classes: int = 4, img_size: int = 28):
        super().__init__()
        # Feature extractor: a small ResNet‑style backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1)
        )
        # The transformer encoder operates on flattened feature patches
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=256, dropout=0.1),
            num_layers=2
        )
        # MLP head that maps the global token to the final four‑dimensional output
        self.mlp = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W) with H=W=28.

        Returns
        -------
        torch.Tensor
            Normalised 4‑dimensional logits of shape (B, 4).
        """
        feats = self.features(x)                     # (B, 64, H', W')
        seq = feats.flatten(2).transpose(1, 2)        # (B, seq_len, 64)
        transformed = self.transformer(seq)          # (B, seq_len, 64)
        cls_token = transformed.mean(dim=1)          # (B, 64)
        out = self.mlp(cls_token)                    # (B, 4)
        return self.norm(out)

__all__ = ["QFCModelExtended"]
