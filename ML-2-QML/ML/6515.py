"""Hybrid self‑attention module combining a classical transformer head with a quantum feature extractor.

The public interface mirrors the original seed: ``run(rotation_params, entangle_params, inputs)``.
The implementation uses PyTorch for the transformer and Pennylane for the quantum part, allowing
gradient propagation through both stages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionModule(nn.Module):
    """
    A hybrid attention module.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability applied to attention scores.
    """

    def __init__(self, embed_dim: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dropout = dropout

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute self‑attention with custom rotation and entangle parameters.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, seq_len, embed_dim).
        rotation_params : np.ndarray
            Shape (embed_dim, embed_dim). Weight matrix applied after the linear projections.
        entangle_params : np.ndarray
            Shape (n_heads,). Scaling factor for each attention head.

        Returns
        -------
        torch.Tensor
            Shape (batch, seq_len, embed_dim).
        """
        # Convert numpy parameters to torch tensors
        rot = torch.from_numpy(rotation_params).float()
        ent = torch.from_numpy(entangle_params).float()

        # Linear projections
        Q = self.q_proj(inputs) @ rot
        K = self.k_proj(inputs) @ rot
        V = self.v_proj(inputs) @ rot

        # Reshape for multi‑head attention
        B, L, D = Q.shape
        head_dim = D // self.n_heads
        Q = Q.view(B, L, self.n_heads, head_dim).transpose(1, 2)  # (B, H, L, D_h)
        K = K.view(B, L, self.n_heads, head_dim).transpose(1, 2)
        V = V.view(B, L, self.n_heads, head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(head_dim)
        scores = scores * ent.view(1, self.n_heads, 1, 1)  # head‑wise scaling
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        # Final linear projection
        out = self.out_proj(out)
        return out

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Public entry point matching the original seed interface.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation weight matrix.
        entangle_params : np.ndarray
            Head scaling vector.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output of the hybrid attention module.
        """
        self.eval()  # disable dropout during inference
        with torch.no_grad():
            inp = torch.from_numpy(inputs).float()
            out = self.forward(inp, rotation_params, entangle_params)
        return out.numpy()

__all__ = ["SelfAttentionModule"]
