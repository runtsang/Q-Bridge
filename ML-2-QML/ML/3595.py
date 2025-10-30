import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ClassicalSelfAttention(nn.Module):
    """
    Differentiable self‑attention block inspired by the original SelfAttention helper.
    It learns query/key/value projections and produces a weighted sum of the values.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        scores = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, V)

class QuanvolutionHybrid(nn.Module):
    """
    Classical hybrid model that replaces the original quanvolution filter with a
    deeper convolutional backbone and augments it with a learnable self‑attention
    block.  The attention block is fully differentiable and can be trained
    end‑to‑end with the rest of the network.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        # Flattened feature dimension: 16 * 7 * 7 = 784
        self.attention = ClassicalSelfAttention(embed_dim=784)
        self.classifier = nn.Linear(784, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)              # [B, 16, 7, 7]
        flat = features.view(features.size(0), -1)  # [B, 784]
        attn_features = self.attention(flat)   # [B, 784]
        logits = self.classifier(attn_features)
        return F.log_softmax(logits, dim=-1)
