import torch
import torch.nn as nn
import numpy as np

class SelfAttentionModule(nn.Module):
    """
    Multi‑head self‑attention with dropout, mirroring the original API.
    The ``run`` method keeps the seed interface (rotation_params,
    entangle_params, inputs) while the ``forward`` method makes the
    module compatible with PyTorch pipelines.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Projection matrices for Q, K, V
        self.W_q = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.W_k = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.W_v = nn.Parameter(torch.empty(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x):
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)

    def _merge_heads(self, x):
        batch, heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch, seq_len, heads * head_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Standard PyTorch forward pass.
        """
        q = torch.matmul(inputs, self.W_q)
        k = torch.matmul(inputs, self.W_k)
        v = torch.matmul(inputs, self.W_v)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = self._merge_heads(attn_output)
        return self.out_proj(attn_output)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Compatibility wrapper that uses the original seed API.
        * ``rotation_params``: flattened weight matrix for Q, K, V stacked as
          ``[W_q, W_k, W_v]``.
        * ``entangle_params``: unused in the classical version but accepted for
          API symmetry; can be interpreted as dropout probabilities.
        """
        # Reshape and load weights
        w_q = rotation_params[:self.embed_dim * self.embed_dim].reshape(self.embed_dim, self.embed_dim)
        w_k = rotation_params[self.embed_dim * self.embed_dim:
                              2 * self.embed_dim * self.embed_dim].reshape(self.embed_dim, self.embed_dim)
        w_v = rotation_params[2 * self.embed_dim * self.embed_dim:].reshape(self.embed_dim, self.embed_dim)

        self.W_q.data = torch.from_numpy(w_q)
        self.W_k.data = torch.from_numpy(w_k)
        self.W_v.data = torch.from_numpy(w_v)

        # Treat entangle_params as dropout probabilities per head
        probs = torch.from_numpy(entangle_params).float()
        if probs.numel() == self.num_heads:
            self.dropout.p = probs.mean().item()

        inp = torch.from_numpy(inputs).float()
        out = self.forward(inp)
        return out.detach().cpu().numpy()

__all__ = ["SelfAttentionModule"]
