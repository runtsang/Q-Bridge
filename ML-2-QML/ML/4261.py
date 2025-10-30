"""Combined classical model: quanvolution filter → self‑attention → regression."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuanvolutionFilter(nn.Module):
    """Standard 2x2 convolutional filter producing 4 feature maps."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: (batch, 4, 14*14)
        return self.conv(x).view(x.size(0), 4, -1)

class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention block over the flattened feature map."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(x.size(-1)), dim=-1)
        return scores @ v

class EstimatorNN(nn.Module):
    """Tiny fully‑connected regressor."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * 14 * 14, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

class Quanvolution__gen261(nn.Module):
    """Integrated quanvolution + attention + regression pipeline."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.attn    = ClassicalSelfAttention(embed_dim=4)
        self.regressor = EstimatorNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        features = self.qfilter(x)                     # (batch, 4, 14*14)
        seq_len = features.size(-1)
        features = features.permute(0, 2, 1)            # (batch, seq_len, 4)
        attn_out = self.attn(features)                 # (batch, seq_len, 4)
        attn_out = attn_out.permute(0, 2, 1).contiguous()   # (batch, 4, seq_len)
        flat = attn_out.view(x.size(0), -1)             # (batch, 4*seq_len)
        logits = self.regressor(flat)
        return logits

__all__ = ["Quanvolution__gen261"]
