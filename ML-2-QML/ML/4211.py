"""Hybrid self-attention module with classical and quantum-capable heads.

The module defines a SelfAttentionHybrid class that implements a multi-head
self-attention block followed by a differentiable sigmoid head.  The head
uses a custom autograd function (HybridFunction) to emulate a quantum
expectation layer while remaining fully classical.  The design mirrors
the binary‑classification network from the reference pair but replaces
convolutional layers with a self‑attention mechanism, making it suitable
for sequence or image‑patch inputs.

The class is fully importable and can be dropped into any PyTorch
pipeline.  It exposes a ``forward`` method that returns logits and a
``predict`` helper that produces class probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid with a learnable shift.

    Mimics the quantum expectation head from the reference pair but
    remains purely classical.  The shift can be tuned during training.
    """

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


class SelfAttentionHybrid(nn.Module):
    """Multi‑head self‑attention with an optional quantum‑style head.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability on the attention weights.
    shift : float, default 0.0
        Shift applied in the HybridFunction.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.classifier = nn.Linear(embed_dim, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, 1).
        """
        B, N, _ = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, self.embed_dim)
        out = self.out_proj(context).sum(dim=1)  # aggregate over sequence

        logits = self.classifier(out)
        probs = HybridFunction.apply(logits.squeeze(-1), self.shift)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return binary probabilities."""
        return self.forward(x)

    def estimate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple regression head: linear mapping from the aggregated
        attention output to a scalar value.  Useful for regression
        tasks where the quantum head is replaced by a classical one.
        """
        B, N, _ = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, self.embed_dim)
        out = self.out_proj(context).sum(dim=1)
        return out.mean(dim=-1)  # return scalar per batch item

__all__ = ["SelfAttentionHybrid", "HybridFunction"]
