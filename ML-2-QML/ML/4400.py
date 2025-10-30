import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Classical self‑attention block inspired by the SelfAttention seed."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable rotation and entangle matrices
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = x @ self.rotation
        key   = x @ self.entangle
        scores = F.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return scores @ x

class QCNNModel(nn.Module):
    """Convolution‑style network mirroring the QCNN seed."""
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

class HybridSamplerQNNModel(nn.Module):
    """Hybrid sampler that combines attention, convolution, and a final linear layer."""
    def __init__(self, input_dim: int = 2, embed_dim: int = 4):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim)
        self.cnn = QCNNModel()
        self.fc = nn.Linear(1, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_dim)
        att = self.attention(x)
        cnn_out = self.cnn(att)
        logits = self.fc(cnn_out)
        return F.softmax(logits, dim=-1)

def HybridSamplerQNN() -> HybridSamplerQNNModel:
    """Factory returning a fully‑connected hybrid sampler."""
    return HybridSamplerQNNModel()

__all__ = ["HybridSamplerQNN"]
