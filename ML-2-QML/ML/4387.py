"""Hybrid self‑attention module that fuses classical attention, a neural sampler, a sequence LSTM and a quantum‑style kernel.

The module can be instantiated in three modes:
* ``use_lstm`` – attaches a classical LSTM to the attention scores to capture temporal context.
* ``use_sampler`` – replaces the soft‑max with a learnable sampler network.
* ``use_quantum_kernel`` – replaces the dot‑product with an RBF kernel that mimics a quantum kernel.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerModule(nn.Module):
    """Simple two‑layer MLP that outputs a probability distribution."""
    def __init__(self, input_dim: int = 4, hidden_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(x), dim=-1)

class Kernel(nn.Module):
    """Classical RBF kernel that emulates a quantum kernel."""
    def __init__(self, gamma: float = 0.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class HybridSelfAttention(nn.Module):
    """Hybrid self‑attention that optionally uses an LSTM, a sampler and a quantum‑style kernel."""
    def __init__(
        self,
        embed_dim: int,
        n_qubits: int = 4,
        use_lstm: bool = False,
        use_sampler: bool = False,
        use_quantum_kernel: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.use_lstm = use_lstm
        self.use_sampler = use_sampler
        self.use_quantum_kernel = use_quantum_kernel

        # Linear projections for query, key, value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        if self.use_sampler:
            self.sampler = SamplerModule(input_dim=embed_dim)
        if self.use_lstm:
            self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        if self.use_quantum_kernel:
            self.kernel = Kernel(gamma=0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape (seq_len, batch, embed_dim)
        Returns:
            Tensor of shape (seq_len, batch, embed_dim)
        """
        queries = self.query_proj(inputs)   # (seq_len, batch, embed_dim)
        keys    = self.key_proj(inputs)
        values  = self.value_proj(inputs)

        seq_len, batch, _ = inputs.shape
        # Compute similarity matrix
        if self.use_quantum_kernel:
            # Use RBF kernel as a stand‑in for a quantum kernel
            sim = torch.zeros(seq_len, seq_len, batch, device=inputs.device)
            for i in range(seq_len):
                for j in range(seq_len):
                    sim[i, j] = self.kernel(queries[i], keys[j]).squeeze(-1)
            sim = sim / sim.sum(dim=1, keepdim=True)  # normalize
        else:
            # Standard dot‑product attention
            scores = torch.einsum('tbi,tbj->tbj', queries, keys) / np.sqrt(self.embed_dim)
            sim = F.softmax(scores, dim=-1)

        # Optionally replace soft‑max with a sampler
        if self.use_sampler:
            # Reshape to (batch, seq_len, seq_len) for per‑batch sampling
            sim = sim.permute(2, 0, 1)  # (batch, seq_len, seq_len)
            sim_flat = sim.reshape(batch * seq_len, seq_len)
            sampled = self.sampler(sim_flat)
            sim = sampled.reshape(batch, seq_len, seq_len).permute(1, 2, 0)

        # Compute weighted sum of values
        attn_output = torch.einsum('tbj,tbj->tbi', sim, values)

        # Optionally add LSTM temporal processing
        if self.use_lstm:
            attn_output, _ = self.lstm(attn_output)

        return attn_output

__all__ = ["HybridSelfAttention"]
