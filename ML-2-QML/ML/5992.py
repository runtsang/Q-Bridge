import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention module with optional hybrid quantum scoring.
    """

    def __init__(self, embed_dim: int, heads: int = 1, dropout: float = 0.0, use_quantum: bool = False):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the input embeddings.
        heads : int, default=1
            Number of attention heads.
        dropout : float, default=0.0
            Dropout probability applied to attention weights.
        use_quantum : bool, default=False
            If True, attention scores are computed by a quantum circuit.
        """
        super().__init__()
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.use_quantum = use_quantum

        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        if self.use_quantum:
            # Instantiate the quantum attention module
            from.quantum_attention import QuantumSelfAttention
            self.quantum_attention = QuantumSelfAttention(num_qubits=self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        B, N, _ = x.size()

        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, N, 3 * embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi‑head attention
        q = q.reshape(B, N, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        k = k.reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        if self.use_quantum:
            scores = []
            for b in range(B):
                # Flatten Q and K to pass to the quantum circuit
                q_np = q[b].reshape(self.heads * N, self.head_dim).cpu().numpy()
                k_np = k[b].reshape(self.heads * N, self.head_dim).cpu().numpy()
                attn_matrix = self.quantum_attention.run(k_np, q_np)  # (N, N)
                scores.append(torch.tensor(attn_matrix, dtype=x.dtype, device=x.device))
            scores = torch.stack(scores)  # (B, N, N)
            scores = scores.unsqueeze(1).expand(-1, self.heads, -1, -1)  # (B, heads, N, N)
        else:
            # Classical scaled dot‑product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, heads, N, N)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        out = torch.matmul(attn, v)  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().reshape(B, N, self.embed_dim)
        out = self.out_proj(out)
        return out
