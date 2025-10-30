import torch
import torch.nn as nn
import numpy as np
from qml_attention import QuantumAttention

class SelfAttentionHybrid(nn.Module):
    """
    Hybrid self‑attention module that blends a quantum‑generated attention matrix
    with a classical transformer‑style weighted sum. The quantum part can be
    trained via gradient‑based optimisation by back‑propagating through the
    parameterised circuit (when using a differentiable simulator).
    """

    def __init__(self, embed_dim: int, n_heads: int = 1, n_tokens: int = 4,
                 rotation_shape: tuple = (4, 3), entangle_shape: tuple = (3,)):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_tokens = n_tokens
        self.rotation_shape = rotation_shape
        self.entangle_shape = entangle_shape

        # Classical linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Quantum attention generator
        self.quantum_attn = QuantumAttention(n_qubits=n_tokens)

        # Learnable parameters for the quantum circuit
        self.rotation_params = nn.Parameter(torch.randn(*rotation_shape))
        self.entangle_params = nn.Parameter(torch.randn(*entangle_shape))

    def forward(self, x: torch.Tensor):
        """
        x: Tensor of shape (batch, seq_len, embed_dim)
        """
        batch, seq_len, _ = x.size()
        assert seq_len == self.n_tokens, "Sequence length must match number of qubits"

        q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Compute query-key scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim)

        # Generate quantum attention matrix
        rotation_np = self.rotation_params.detach().cpu().numpy()
        entangle_np = self.entangle_params.detach().cpu().numpy()
        attn_matrix = self.quantum_attn.get_attention_matrix(rotation_np, entangle_np)
        attn_matrix = torch.tensor(attn_matrix, dtype=x.dtype, device=x.device)

        # Combine quantum attention with classical scores
        combined_scores = scores * attn_matrix  # elementwise multiplication

        attn_weights = torch.softmax(combined_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        return out

__all__ = ["SelfAttentionHybrid"]
