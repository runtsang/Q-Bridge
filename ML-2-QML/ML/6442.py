from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionHybrid:
    """
    Classical self‑attention module with trainable query/key/value matrices.
    The API is compatible with the original seed but extended to support
    GPU acceleration and optional post‑processing.
    """

    def __init__(self, embed_dim: int, device: str | None = None):
        self.embed_dim = embed_dim
        self.device = torch.device(device or "cpu")
        # Trainable linear layers for query, key and value
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        # Move to device
        self.to(self.device)

    def to(self, device):
        self.device = torch.device(device)
        self.query.to(device)
        self.key.to(device)
        self.value.to(device)

    def run(self, rotation_params: torch.Tensor, entangle_params: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute self‑attention output.
        rotation_params and entangle_params are accepted for API compatibility
        but are not used directly in the classical block; they can be used
        for hybrid training if needed.
        """
        # Ensure inputs are on the correct device
        inputs = inputs.to(self.device)
        # Compute query, key, value
        Q = self.query(inputs)  # (batch, seq, dim)
        K = self.key(inputs)
        V = self.value(inputs)
        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output

    def get_attention_weights(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Return the attention weight matrix for given inputs.
        """
        Q = self.query(inputs)
        K = self.key(inputs)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        return F.softmax(scores, dim=-1)

__all__ = ["SelfAttentionHybrid"]
