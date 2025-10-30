"""ML implementation of a hybrid convolutional network with classical self‑attention
and a lightweight sigmoid head.  The design mirrors the quantum baseline but replaces
all quantum primitives with high‑performance PyTorch operations.

Shared class name: HybridAttentionQCNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalSelfAttention(nn.Module):
    """Dot‑product self‑attention implemented purely in PyTorch."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Attention‑weighted representation of shape (batch, seq_len, embed_dim).
        """
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5), dim=-1)
        return torch.matmul(scores, v)

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head that mimics a quantum expectation."""
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
    """Simple dense head that replaces the quantum circuit."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class HybridAttentionQCNet(nn.Module):
    """CNN + self‑attention + hybrid sigmoid head."""
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1   = nn.Linear(55815, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, embed_dim)
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        self.hybrid = Hybrid(embed_dim, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv backbone
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))

        # Self‑attention on the last hidden dimension
        x = self.fc3(x)                           # (batch, embed_dim)
        x = x.view(x.size(0), 1, -1)              # (batch, seq_len=1, embed_dim)
        attn_out = self.attention(x).squeeze(1)   # (batch, embed_dim)

        # Hybrid sigmoid head
        probs = self.hybrid(attn_out)              # (batch, 1)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["ClassicalSelfAttention", "HybridFunction", "Hybrid", "HybridAttentionQCNet"]
