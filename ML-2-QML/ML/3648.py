"""Hybrid attention network combining classical self‑attention with quantum-inspired head.

This module defines a purely classical neural network that
uses a learnable self‑attention block before a small linear head.
The architecture mirrors the quantum hybrid model but replaces
the quantum circuit with a differentiable sigmoid head.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalSelfAttention(nn.Module):
    """Simple scaled‑dot‑product self‑attention implemented in PyTorch."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable projection matrices
        self.Wq = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Wk = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Wv = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, embed_dim)
        q = inputs @ self.Wq
        k = inputs @ self.Wk
        v = inputs @ self.Wv
        scores = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.embed_dim), dim=-1)
        output = torch.matmul(scores, v)
        return output.mean(-1)

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head that mimics a quantum expectation layer."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Classical linear head with a sigmoid activation."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class HybridAttentionNet(nn.Module):
    """CNN → linear embedding → self‑attention → fully‑connected → hybrid head."""
    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Linear embedding to 4‑dimensional space for attention
        self.linear_embed = nn.Linear(55815, 4)

        # Self‑attention block
        self.attention = ClassicalSelfAttention(embed_dim=4)

        # Fully‑connected layers
        self.fc1 = nn.Linear(4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Hybrid head
        self.hybrid = Hybrid(1, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        x = self.linear_embed(x)          # (batch, 4)
        x = self.attention(x)            # (batch, 4)
        x = torch.flatten(x, 1)           # (batch, 4)

        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        probabilities = self.hybrid(x)
        return torch.cat((probabilities, 1 - probabilities), dim=-1)

__all__ = ["ClassicalSelfAttention", "HybridFunction", "Hybrid", "HybridAttentionNet"]
