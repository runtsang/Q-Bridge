import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalQuanvolutionFilter(nn.Module):
    """Classical convolution inspired by the original quanvolution filter."""
    def __init__(self, in_channels=1, out_channels=4, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x.view(x.size(0), -1)

class ClassicalSelfAttention(nn.Module):
    """Learnable self‑attention block implemented with linear layers."""
    def __init__(self, embed_dim=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

class QuanvolutionAttentionModel(nn.Module):
    """Classical model that stacks a quanvolution filter, self‑attention, and a classifier head."""
    def __init__(self, num_classes=10, embed_dim=4):
        super().__init__()
        self.filter = ClassicalQuanvolutionFilter()
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        self.head = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.filter(x)
        x = self.attention(x)
        logits = self.head(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionAttentionModel"]
