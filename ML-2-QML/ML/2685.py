"""Hybrid classical self‑attention + QCNN inspired network.

This module defines a PyTorch model that first applies a multi‑head
self‑attention block mimicking the rotation/entanglement pattern of
the quantum self‑attention circuit, and then processes the attended
representation through a stack of fully‑connected layers that emulate
the convolution and pooling stages of a QCNN.  The design allows
direct comparison with the quantum counterpart while retaining
classical efficiency.
"""

import torch
from torch import nn
import numpy as np


class HybridSelfAttentionQCNNModel(nn.Module):
    """
    A PyTorch model that combines multi‑head self‑attention with
    QCNN‑style fully‑connected layers.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 16,
        num_heads: int = 4,
        hidden_sizes: list[int] | None = None,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input feature vector.
        embed_dim : int, optional
            Dimensionality of the attention embedding.
        num_heads : int, optional
            Number of attention heads.
        hidden_sizes : list[int] | None, optional
            Sizes of the successive QCNN‑style layers.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(input_dim, embed_dim)
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, embed_dim)

        # QCNN‑style fully‑connected stack
        if hidden_sizes is None:
            hidden_sizes = [32, 32, 16, 8]
        layers = []
        in_dim = embed_dim
        for out_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
            in_dim = out_dim
        self.qcnn_layers = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch,).
        """
        # ----- Multi‑head self‑attention -----
        q = self.q_proj(x)          # (batch, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        batch = q.shape[0]

        # Reshape for token‑wise attention across the embedding dimension
        q = q.view(batch, self.embed_dim, 1)
        k = k.view(batch, self.embed_dim, 1)
        v = v.view(batch, self.embed_dim, 1)

        # Attention scores and weighted sum
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, embed_dim, embed_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v).squeeze(-1)     # (batch, embed_dim)

        # ----- QCNN‑style layers -----
        out = self.qcnn_layers(attn_output)
        out = torch.sigmoid(self.head(out))
        return out.squeeze(-1)


def HybridSelfAttentionQCNN() -> HybridSelfAttentionQCNNModel:
    """
    Factory that returns a ready‑to‑use instance of the hybrid model.
    """
    return HybridSelfAttentionQCNNModel(input_dim=8)
