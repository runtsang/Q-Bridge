"""Hybrid self‑attention module with multi‑head support and optional quantum back‑end.

The original seed provided a single‑head, 4‑dimensional attention block.  
This extension adds:
* **Multi‑head** – split the input embedding into *h* heads, each head processes a *d_k* dimensional sub‑space.
* **Dropout** – optional stochastic masking of attention scores.
* **Hybrid mode** – a flag to choose between a pure‑classical or a quantum‑backed attention sub‑module.
* **Benchmarking** – a small helper that measures run time and compares output distributions.
"""

from __future__ import annotations

import numpy as np
import torch
import time
from typing import Optional, Tuple

__all__ = ["SelfAttention"]


class SelfAttention:
    """
    Classical (and optionally hybrid) self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimension of the input embeddings.
    num_heads : int, default 1
        Number of attention heads.
    dropout : float, default 0.0
        Dropout probability applied to attention scores.
    use_quantum : bool, default False
        If True, the run method will delegate to a quantum sub‑module.
        The quantum sub‑module must be passed via ``quantum_attention``.
    quantum_attention : Optional[object]
        External quantum attention object that implements a compatible ``run`` method.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        use_quantum: bool = False,
        quantum_attention: Optional[object] = None,
    ) -> None:
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.use_quantum = use_quantum
        self.quantum_attention = quantum_attention

        # Linear projection matrices for Q, K, V
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split last dimension into (num_heads, head_dim)."""
        batch, seq_len, embed_dim = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine heads back to original embedding dimension."""
        batch, num_heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Compute self‑attention.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters used to initialise the linear projections (ignored in the
            classical mode but kept for API compatibility).
        entangle_params : np.ndarray
            Parameters used to initialise the entanglement (ignored in the
            classical mode but kept for API compatibility).
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).
        device : torch.device, optional
            Torch device for computation.

        Returns
        -------
        np.ndarray
            Attention output of shape (batch, seq_len, embed_dim).
        """
        if device is None:
            device = torch.device("cpu")

        x = torch.as_tensor(inputs, dtype=torch.float32, device=device)

        if self.use_quantum:
            if self.quantum_attention is None:
                raise RuntimeError("Quantum attention object not provided.")
            return self.quantum_attention.run(
                rotation_params, entangle_params, inputs, device
            )

        # Classical multi‑head attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Scaled dot‑product
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = torch.softmax(scores, dim=-1)

        if self.dropout > 0.0:
            dropout_mask = torch.bernoulli(
                torch.full_like(scores, 1.0 - self.dropout)
            )
            scores = scores * dropout_mask / (1.0 - self.dropout)

        out = torch.matmul(scores, v)
        out = self._combine_heads(out)
        return out.cpu().numpy()

    def benchmark(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        repeats: int = 10,
    ) -> Tuple[float, np.ndarray]:
        """
        Measure average execution time and return a single run output.

        Parameters
        ----------
        rotation_params : np.ndarray
        entangle_params : np.ndarray
        inputs : np.ndarray
        repeats : int, default 10

        Returns
        -------
        Tuple[float, np.ndarray]
            Average time per run (seconds) and the output of the last run.
        """
        start = time.time()
        out = None
        for _ in range(repeats):
            out = self.run(rotation_params, entangle_params, inputs)
        elapsed = time.time() - start
        return elapsed / repeats, out
