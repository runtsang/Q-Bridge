import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionEnhanced(nn.Module):
    """
    A batched, transformer‑style self‑attention module that mirrors the quantum interface.
    Provides learnable linear projections, dropout, and a residual connection.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, rotation_params: torch.Tensor, entangle_params: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch, seq_len, embed_dim)
        rotation_params: (batch, seq_len, embed_dim) – used as a simple mask
        entangle_params: (batch, seq_len, embed_dim) – used as a simple mask
        """
        # apply projection
        Q = self.q_proj(inputs)
        K = self.k_proj(inputs)
        V = self.v_proj(inputs)

        # compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        # weighted sum
        attn_output = torch.matmul(scores, V)

        # residual & output projection
        attn_output = self.out_proj(attn_output)
        return attn_output + inputs

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Compatibility wrapper to match the original interface.
        """
        inp = torch.from_numpy(inputs).float()
        rot = torch.from_numpy(rotation_params).float()
        ent = torch.from_numpy(entangle_params).float()
        out = self.forward(inp, rot, ent)
        return out.detach().numpy()
