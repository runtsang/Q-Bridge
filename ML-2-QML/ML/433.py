import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention__gen062(nn.Module):
    """Hybrid self‑attention combining classical projections with a quantum‑derived mask.

    The module is fully differentiable and can be trained end‑to‑end with
    standard PyTorch optimisers.  The quantum mask is supplied as a
    callable that returns a probability vector of shape (seq_len,).
    """
    def __init__(self, embed_dim: int, seq_len: int, quantum_mask_fn=None):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of input embeddings.
        seq_len : int
            Length of the sequence (number of tokens).
        quantum_mask_fn : callable or None
            Function that takes a batch of parameters and returns a
            probability mask of shape (batch, seq_len).  If None, a
            uniform mask is used.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.quantum_mask_fn = quantum_mask_fn

        # Classical linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Optional linear layer to convert quantum output to mask
        if quantum_mask_fn is not None:
            self.mask_linear = nn.Linear(seq_len, seq_len)

    def forward(self, x: torch.Tensor, quantum_params: torch.Tensor = None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        quantum_params : torch.Tensor, optional
            Parameters to be forwarded to the quantum mask function.

        Returns
        -------
        torch.Tensor
            Output of shape (batch, seq_len, embed_dim).
        """
        # Classical projections
        q = self.q_proj(x)          # (B, L, D)
        k = self.k_proj(x)          # (B, L, D)
        v = self.v_proj(x)          # (B, L, D)

        # Scaled dot‑product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, L, L)
        scores = scores / (self.embed_dim ** 0.5)

        # Quantum‑derived mask (if provided)
        if self.quantum_mask_fn is not None and quantum_params is not None:
            q_mask = self.quantum_mask_fn(quantum_params)  # (B, L)
            # Expand to (B, L, L) and apply element‑wise multiplication
            q_mask = q_mask.unsqueeze(2)  # (B, L, 1)
            scores = scores * q_mask  # Broadcasting over the last dim

        # Softmax over keys
        attn = F.softmax(scores, dim=-1)  # (B, L, L)

        # Weighted sum of values
        out = torch.matmul(attn, v)  # (B, L, D)
        return out
