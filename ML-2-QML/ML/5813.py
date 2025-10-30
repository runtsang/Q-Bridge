"""
Classical Self‑Attention module with multi‑head, dropout, and training utilities.
Extends the original single‑head implementation by supporting multiple heads,
dropout regularization, and a simple training loop.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionModule:
    """
    Multi‑head self‑attention with optional dropout.
    Parameters:
        embed_dim (int): Dimensionality of input embeddings.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability applied to attention weights.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray) -> torch.Tensor:
        """
        Compute multi‑head self‑attention.
        Args:
            inputs (torch.Tensor): (batch, seq_len, embed_dim)
            rotation_params (np.ndarray): Rotation angles for each head,
                                          shape (num_heads, head_dim, 3)
            entangle_params (np.ndarray): Entanglement angles, shape (num_heads, head_dim)
        Returns:
            torch.Tensor: (batch, seq_len, embed_dim)
        """
        batch, seq_len, _ = inputs.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(inputs)  # (batch, seq_len, 3*embed_dim)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]  # each: (batch, seq_len, num_heads, head_dim)

        # Apply rotation parameters per head (Euler angles -> simplified rotation)
        cos_theta = torch.tensor(np.cos(rotation_params[..., 0]), device=q.device)
        sin_theta = torch.tensor(np.sin(rotation_params[..., 0]), device=q.device)
        q_rot = cos_theta * q + sin_theta * k

        # Compute scaled dot‑product attention per head
        scores = torch.matmul(q_rot, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout_layer(attn)

        out = torch.matmul(attn, v)  # (batch, seq_len, num_heads, head_dim)
        out = out.reshape(batch, seq_len, self.embed_dim)
        out = self.out_proj(out)
        return out

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: torch.nn.modules.loss._Loss) -> float:
        """
        One training step: forward, loss, backward, step.
        Returns the loss value.
        """
        self.train()
        optimizer.zero_grad()
        rotation_params = np.random.uniform(0, 2*np.pi,
                                            size=(self.num_heads, self.head_dim, 3))
        entangle_params = np.random.uniform(0, 2*np.pi,
                                            size=(self.num_heads, self.head_dim))
        outputs = self.forward(inputs, rotation_params, entangle_params)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

__all__ = ["SelfAttentionModule"]
