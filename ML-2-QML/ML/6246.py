"""Hybrid attention‑based binary classifier for classical training.

The network keeps the convolutional backbone of the original
`ClassicalQuantumBinaryClassification` but replaces the quantum head
with a differentiable sigmoid head.  A lightweight self‑attention
module is inserted after the convolutional feature extractor to
capture long‑range dependencies.  The implementation is fully
PyTorch‑based and can be trained on commodity GPUs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention:
    """Simple self‑attention block operating on a 1‑D feature vector."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed
        # Random projection matrices are fixed – they act as a
        # lightweight, learnable attention kernel.
        self.q_proj = np.random.randn(embed_dim, embed_dim)
        self.k_proj = np.random.randn(embed_dim, embed_dim)
        self.v_proj = np.random.randn(embed_dim, embed_dim)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        q = inputs @ self.q_proj
        k = inputs @ self.k_proj
        v = inputs @ self.v_proj
        scores = np.exp(q @ k.T / np.sqrt(self.embed_dim))
        scores /= scores.sum(axis=-1, keepdims=True)
        return scores @ v

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head with an optional bias shift."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        return torch.sigmoid(inputs + shift)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (inputs,) = ctx.saved_tensors
        sigmoid = torch.sigmoid(inputs)
        return grad_output * sigmoid * (1 - sigmoid), None

class Hybrid(nn.Module):
    """Dense layer followed by a sigmoid head."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs)
        return HybridFunction.apply(logits, self.shift)

class HybridAttentionQCNet(nn.Module):
    """Convolutional network with a self‑attention module and a quantum‑inspired head."""
    def __init__(self, attention_dim: int = 4):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop_conv = nn.Dropout2d(p=0.25)

        # Attention head
        self.attention_fc = nn.Linear(16 * 7 * 7, attention_dim)
        self.self_attention = ClassicalSelfAttention(embed_dim=attention_dim)

        # Classification head
        self.fc1 = nn.Linear(attention_dim, 120)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(1, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop_conv(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop_conv(x)

        # Flatten and attention
        x = torch.flatten(x, 1)
        attn_features = self.attention_fc(x)
        attn_output = torch.from_numpy(self.self_attention.run(attn_features.detach().cpu().numpy()))
        x = attn_output

        # Classification
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridAttentionQCNet", "Hybrid", "HybridFunction", "ClassicalSelfAttention"]
