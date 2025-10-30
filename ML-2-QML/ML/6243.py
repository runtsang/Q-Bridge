"""SelfAttention: Classical attention module with batched support and training utilities.

The class exposes:
- forward: compute scaled‑dot‑product attention with optional multi‑head.
- compute_attention: convenience wrapper for NumPy inputs/outputs.
- fit: lightweight parameter training using Adam and MSE loss.
"""

import numpy as np
import torch
from torch import nn, optim


class SelfAttention(nn.Module):
    """
    Classical self‑attention module mirroring the quantum interface.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        # Parameter matrices for query, key, value
        self.w_q = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.w_k = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.w_v = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor) -> torch.Tensor:
        """
        Compute attention output.

        Parameters
        ----------
        inputs : torch.Tensor of shape (batch, seq_len, embed_dim)
        rotation_params : torch.Tensor of shape (embed_dim, embed_dim)
        entangle_params : torch.Tensor of shape (embed_dim, embed_dim)
        """
        # Apply rotation and entangle matrices as linear transforms
        query = torch.matmul(inputs, rotation_params)
        key = torch.matmul(inputs, entangle_params)
        value = inputs

        # Reshape for multi‑head
        batch, seq_len, _ = query.shape
        query = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, value)
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return output

    def compute_attention(self,
                          inputs: np.ndarray,
                          rotation_params: np.ndarray,
                          entangle_params: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper that accepts NumPy arrays and returns NumPy output.
        """
        inputs_t = torch.tensor(inputs, dtype=torch.float32, requires_grad=False)
        rotation_t = torch.tensor(rotation_params, dtype=torch.float32)
        entangle_t = torch.tensor(entangle_params, dtype=torch.float32)
        with torch.no_grad():
            out_t = self.forward(inputs_t, rotation_t, entangle_t)
        return out_t.numpy()

    def fit(self,
            inputs: np.ndarray,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            lr: float = 1e-3,
            epochs: int = 200,
            verbose: bool = False) -> None:
        """
        Train the rotation and entangle matrices to match a target attention output.
        """
        inputs_t = torch.tensor(inputs, dtype=torch.float32)
        target = self.forward(inputs_t,
                              torch.tensor(rotation_params, dtype=torch.float32),
                              torch.tensor(entangle_params, dtype=torch.float32))
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(inputs_t, self.w_q, self.w_k)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:04d} loss={loss.item():.6f}")
