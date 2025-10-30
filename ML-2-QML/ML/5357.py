from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Standard scaled dot‑product self‑attention implemented in PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.out_proj(attn_output)

class QuanvolutionFilter(nn.Module):
    """Classical emulation of a 2‑qubit quantum kernel applied to 2×2 patches."""
    def __init__(self, patch_size: int = 2, out_features: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.out_features = out_features
        self.conv = nn.Conv2d(1, out_features, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QCNNModel(nn.Module):
    """A lightweight feed‑forward chain that mirrors a quantum convolution‑pooling stack."""
    def __init__(self):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridAttentionNetwork(nn.Module):
    """
    Hybrid classical‑quantum attention network.

    * Classical self‑attention captures global dependencies.
    * QuanvolutionFilter injects a quantum‑kernel style feature extractor.
    * QCNNModel emulates a quantum convolution‑pooling hierarchy.
    * The optional `use_quantum` flag reserves a slot for a true quantum backend
      (in the accompanying QML module).
    """
    def __init__(self, embed_dim: int = 4, num_heads: int = 1, use_quantum: bool = False):
        super().__init__()
        self.use_quantum = use_quantum
        self.attention = ClassicalSelfAttention(embed_dim, num_heads)
        self.quanvolution = QuanvolutionFilter()
        self.qcnn = QCNNModel()
        if use_quantum:
            # placeholder: the real quantum implementation lives in the QML module
            self.quantum_bridge = nn.Identity()
        else:
            self.quantum_bridge = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, embed_dim) for attention; (B, 1, H, W) for convolution
        attn_out = self.attention(x)
        # Assume a dummy image tensor for the convolutional path
        # In practice the caller should provide an image feature map of shape (B,1,H,W)
        dummy_img = torch.randn(x.size(0), 1, 28, 28, device=x.device)
        quanv_out = self.quanvolution(dummy_img)
        qcnn_out = self.qcnn(quanv_out)
        out = attn_out + qcnn_out
        out = self.quantum_bridge(out)
        return out

__all__ = ["HybridAttentionNetwork"]
