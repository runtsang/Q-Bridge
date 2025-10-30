import torch
from torch import nn
import numpy as np

class SelfAttentionModule(nn.Module):
    """Transformer‑style self‑attention block."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out)

class HybridQCNNSA(nn.Module):
    """Hybrid QCNN with a self‑attention block."""
    def __init__(self, input_dim: int = 8, embed_dim: int = 4):
        super().__init__()
        # Feature map
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        # Convolution layers
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.pool3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Self‑attention
        self.attention = SelfAttentionModule(embed_dim)
        # Classification head
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through QCNN layers
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        # Reshape for attention: (batch, seq_len=1, embed_dim)
        x_att = x.unsqueeze(1)
        x_att = self.attention(x_att)
        x_att = x_att.squeeze(1)
        # Classification
        out = self.head(x_att)
        return torch.sigmoid(out)
