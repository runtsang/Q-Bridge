import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalQuanvolutionFilter(nn.Module):
    """2×2 classical filter with stride 2, analogous to the original."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class ClassicalSelfAttention(nn.Module):
    """Pure‑Python self‑attention block inspired by the quantum interface."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # params: (batch, embed_dim*3) -> split into rotation, entangle, value
        rotation = params[:, :self.embed_dim]
        entangle  = params[:, self.embed_dim:2*self.embed_dim]
        q = self.query_proj(x @ rotation.T)
        k = self.key_proj(x @ entangle.T)
        v = self.value_proj(x)
        scores = F.softmax(q @ k.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

class QCNNBlock(nn.Module):
    """Stack of linear layers emulating a QCNN feature extractor."""
    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))

class QuanvolutionHybrid(nn.Module):
    """Full model: classical quanvolution → self‑attention → QCNN head."""
    def __init__(self, attention_params: torch.Tensor | None = None):
        super().__init__()
        self.qfilter = ClassicalQuanvolutionFilter()
        self.attn = ClassicalSelfAttention(embed_dim=4)
        # feature map size after 28x28 -> 14x14 patches * 4 channels = 784
        self.head = QCNNBlock(input_dim=4*14*14)
        self.attn_params = attention_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qfilter(x)
        if self.attn_params is not None:
            x = self.attn(x, self.attn_params)
        logits = self.head(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["ClassicalQuanvolutionFilter", "ClassicalSelfAttention",
           "QCNNBlock", "QuanvolutionHybrid"]
