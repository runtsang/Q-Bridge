import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Simple dot‑product self‑attention operating on a sequence of embeddings."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, inputs: torch.Tensor,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, seq_len, embed_dim)
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key = inputs @ entangle_params.reshape(self.embed_dim, -1)
        scores = F.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, inputs)

class ClassicalQuanvolutionFilter(nn.Module):
    """2×2 patchwise 2‑D convolution that mimics a quantum kernel."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        return self.conv(x).view(x.size(0), -1)

class HybridQuanvolutionAttentionClassifier(nn.Module):
    """Hybrid classical pipeline: quanvolution → self‑attention → linear head."""
    def __init__(self, num_classes: int = 10, embed_dim: int = 4):
        super().__init__()
        self.qfilter = ClassicalQuanvolutionFilter()
        self.attention = ClassicalSelfAttention(embed_dim)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Phase 1: quanvolution
        feat = self.qfilter(x)  # (batch, 4*14*14)
        # Phase 2: reshape into sequence of embeddings
        seq_len = feat.size(1) // self.attention.embed_dim
        feat_seq = feat.view(-1, seq_len, self.attention.embed_dim)
        # Phase 3: generate random attention parameters
        rot = torch.randn(self.attention.embed_dim, self.attention.embed_dim, device=x.device)
        ent = torch.randn(self.attention.embed_dim, self.attention.embed_dim, device=x.device)
        # Phase 4: apply self‑attention
        attn_feat = self.attention(feat_seq, rot, ent)
        attn_feat = attn_feat.view(-1, 4 * 14 * 14)
        # Phase 5: classification head
        logits = self.linear(attn_feat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionAttentionClassifier"]
