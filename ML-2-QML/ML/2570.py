"""Classical attention‑based binary classifier with hybrid head.

This module defines AttentionHybridQCNet that extends the classical
convolutional architecture with a self‑attention block and a
parameterised sigmoid head.  The design mirrors the quantum
counterpart but remains fully PyTorch‑based, enabling fast CPU
training and easy integration with existing pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Simple multi‑head attention implemented with PyTorch tensors."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = np.sqrt(self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, embed_dim)
        q = x
        k = x
        v = x
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation with a trainable shift."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Dense head that mimics the quantum expectation layer."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class AttentionHybridQCNet(nn.Module):
    """Classical CNN + attention + hybrid sigmoid head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)   # embed_dim for attention
        self.attention = ClassicalSelfAttention(embed_dim=4)
        self.hybrid = Hybrid(in_features=1, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # shape (batch, 4)
        x = self.attention(x)  # attention output shape (batch, 4)
        # Collapse attention output to a scalar for the hybrid head
        x = torch.mean(x, dim=-1, keepdim=True)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["AttentionHybridQCNet"]
