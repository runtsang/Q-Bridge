import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, BatchNorm2d
from torch.nn.functional import unfold

class QuanvolutionFilter(nn.Module):
    """Classical 2‑D convolution with attention‑weighted patch extraction."""
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        conv_kernel: int = 2,
        conv_stride: int = 2,
        patch_size: int = 2,
        patch_stride: int = 2,
    ) -> None:
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=conv_kernel, stride=conv_stride)
        self.bn = BatchNorm2d(out_channels)
        # Attention branch produces a weight map for each spatial location
        self.attn = nn.Sequential(
            Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, H, W)
        x = self.conv(x)
        x = self.bn(x)
        attn_map = self.attn(x)  # (B, out, H', W')
        # Extract 2x2 patches from the feature map
        patches = unfold(
            x,
            kernel_size=self.patch_size,
            stride=self.patch_stride
        )  # (B, out*ps*ps, Npatch)
        # Broadcast attention to the patches
        attn = attn_map.view(x.size(0), -1, 1)  # (B, out*ps*ps, 1)
        patches = patches * attn
        # Flatten patches across channel and spatial dimensions
        patches = patches.permute(0, 2, 1)  # (B, Npatch, out*ps*ps)
        features = patches.reshape(x.size(0), -1)
        return features

class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier that uses the enhanced classical filter."""
    def __init__(self, num_classes: int = 10, in_channels: int = 1) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels=in_channels)
        # After conv stride 2 on 28x28 image -> 14x14 patches, each patch has out*ps*ps = 4*2*2 = 16 features
        self.linear = Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
