import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Simple 2×2 convolution filter that emulates the quanvolution idea."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class SelfAttention(nn.Module):
    """Classic self‑attention built on top of a quanvolution filter."""
    def __init__(self, embed_dim: int = 4, n_filters: int = 4):
        super().__init__()
        self.qfilter = QuanvolutionFilter(out_channels=n_filters)
        # linear projections for query/key/value
        self.w_q = nn.Linear(n_filters * 14 * 14, embed_dim, bias=False)
        self.w_k = nn.Linear(n_filters * 14 * 14, embed_dim, bias=False)
        self.w_v = nn.Linear(n_filters * 14 * 14, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input image batch of shape (B, 1, 28, 28).

        Returns
        -------
        Tensor
            Attention‑weighted representation of shape (B, embed_dim).
        """
        B = x.shape[0]
        features = self.qfilter(x)  # (B, 4*14*14)
        # linear projections
        Q = self.w_q(features)      # (B, embed_dim)
        K = self.w_k(features)      # (B, embed_dim)
        V = self.w_v(features)      # (B, embed_dim)
        # scaled dot‑product attention
        scores = torch.softmax(Q @ K.transpose(-2, -1) / (embed_dim ** 0.5), dim=-1)
        out = scores @ V              # (B, embed_dim)
        return out

__all__ = ["SelfAttention", "QuanvolutionFilter"]
