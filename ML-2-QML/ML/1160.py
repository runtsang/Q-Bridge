import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """
    Classical quanvolution filter that replaces the fixed 2×2 convolution
    with a learnable 2×2 kernel and a multi‑head self‑attention block
    operating on the extracted patches.
    """

    def __init__(self, in_channels=1, out_channels=4, patch_size=2, heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=patch_size, stride=patch_size)
        self.attn = nn.MultiheadAttention(embed_dim=out_channels,
                                          num_heads=heads,
                                          batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        batch, channels, h, w = features.shape
        features = features.view(batch, channels, -1).transpose(1, 2)
        attn_output, _ = self.attn(features, features, features)
        attn_output = attn_output.mean(dim=1)
        return attn_output

class QuanvolutionClassifier(nn.Module):
    """
    Hybrid classifier that uses the quanvolution filter
    followed by a linear head for 10‑class classification.
    """

    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels=in_channels)
        self.linear = nn.Linear(self.qfilter.conv.out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
