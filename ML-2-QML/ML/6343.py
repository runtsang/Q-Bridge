"""
Extended classical quanvolution model with optional residual and attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionExtended(nn.Module):
    """
    Classical quanvolution model with optional residual connection and channel-wise attention.
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 4,
                 kernel_size: int = 2,
                 stride: int = 2,
                 use_residual: bool = False,
                 use_attention: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.use_residual = use_residual
        if use_residual:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.use_attention = use_attention
        if use_attention:
            # Simple channel-wise attention: global average pooling + 1-layer MLP
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, out_channels // 2),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels // 2, out_channels),
                nn.Sigmoid()
            )
        # Compute output feature size: after conv with stride 2 on 28x28 -> 14x14 patches
        # Each patch outputs out_channels features
        self.feature_dim = out_channels * 14 * 14
        self.linear = nn.Linear(self.feature_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28)

        Returns:
            Log-softmax logits of shape (batch, 10)
        """
        features = self.conv(x)
        if self.use_residual:
            residual = self.res_conv(x)
            features = features + residual
        if self.use_attention:
            attn = self.attention(features)
            features = features * attn.unsqueeze(-1).unsqueeze(-1)
        # Flatten
        features = features.view(x.size(0), -1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionExtended"]
