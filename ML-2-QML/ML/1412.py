import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    """
    Hybrid self‑attention module supporting classical scaled dot‑product attention
    and a placeholder for quantum‑augmented score computation.  The class
    integrates multi‑head attention, dropout, and a small feed‑forward network,
    making it suitable for downstream tasks such as classification or
    sequence modeling.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of token embeddings.
    num_heads : int, default 1
        Number of attention heads.
    use_quantum : bool, default False
        If True, the attention scores are perturbed by a quantum‑derived noise
        term (placeholder for a real VQC).
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, use_quantum: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_quantum = use_quantum

        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Feed‑forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Dropout on attention weights
        self.dropout = nn.Dropout(p=0.1)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape and permute to (batch, num_heads, seq_len, head_dim).
        """
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of _split_heads.
        """
        batch, num_heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch, seq_len, num_heads * head_dim)

    def _classical_attention(self, q, k, v):
        """
        Compute scaled dot‑product attention.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, v)

    def _quantum_attention(self, q, k, v, rotation_params, entangle_params):
        """
        Placeholder for a quantum‑augmented attention mechanism.
        The quantum circuit would normally produce a similarity score
        between each query‑key pair.  Here we add a small random perturbation
        to the classical scores to mimic quantum noise.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        noise = torch.randn_like(scores) * 0.05
        scores = scores + noise
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, v)

    def run(self, inputs: torch.Tensor,
            rotation_params: np.ndarray = None,
            entangle_params: np.ndarray = None) -> torch.Tensor:
        """
        Forward pass of the self‑attention block.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, seq_len, embed_dim).
        rotation_params, entangle_params : np.ndarray, optional
            Parameters for the quantum circuit if `use_quantum` is True.
        """
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.use_quantum and rotation_params is not None and entangle_params is not None:
            attn_output = self._quantum_attention(q, k, v, rotation_params, entangle_params)
        else:
            attn_output = self._classical_attention(q, k, v)

        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        attn_output = self.ffn(attn_output)
        return attn_output

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)

__all__ = ["SelfAttention"]
