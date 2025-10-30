"""Classical hybrid self‑attention binary classifier.

This module implements a convolutional backbone followed by a
classical self‑attention block and a dense head.  The architecture
mirrors the quantum counterpart, enabling direct ablation studies
on the impact of the quantum layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Dot‑product self‑attention over the feature dimension."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq, embed_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, v)

class HybridFunction(nn.Module):
    """Simple sigmoid head that emulates a quantum expectation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

class Hybrid(nn.Module):
    """Linear layer followed by a sigmoid activation."""
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.activation = HybridFunction()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))

class HybridSelfAttentionQCNet(nn.Module):
    """End‑to‑end binary classifier combining convolution, classical
    self‑attention and a dense head.  The architecture is identical
    to the quantum version except for the head.
    """
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Dummy forward to compute feature dimension
        dummy = torch.zeros(1, 3, 32, 32)
        dummy = self._forward_features(dummy)
        feat_dim = dummy.shape[-1]

        self.self_attn = ClassicalSelfAttention(embed_dim=embed_dim)
        self.fc1 = nn.Linear(feat_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(self.fc3.out_features)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        seq = x.shape[-1] // self.self_attn.embed_dim
        x = x.view(-1, seq, self.self_attn.embed_dim)
        x = self.self_attn(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        out = self.hybrid(x)
        return torch.cat((out, 1 - out), dim=-1)

__all__ = ["HybridSelfAttentionQCNet"]
