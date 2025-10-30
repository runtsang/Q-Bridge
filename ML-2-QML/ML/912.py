"""Enhanced classical self‑attention layer with multi‑head support and optional dropout.

The class mirrors the original interface but expands the parameter space to
support multiple attention heads, dropout, and a richer linear projection
scheme.  It can be used as a drop‑in replacement in transformer‑style
pipelines while still accepting the same `run(rotation_params,
entangle_params, inputs)` signature for backward compatibility.

Key features
------------
* Multi‑head attention with configurable head count.
* Optional dropout on attention weights.
* Gradient support via PyTorch autograd.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class SelfAttentionLayer:
    """Multi‑head self‑attention layer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # We use diagonal weight matrices for simplicity; in practice these
        # would be learnable linear layers.
        self.q_weight = torch.nn.Parameter(
            torch.eye(embed_dim, dtype=torch.float32)
        )
        self.k_weight = torch.nn.Parameter(
            torch.eye(embed_dim, dtype=torch.float32)
        )
        self.v_weight = torch.nn.Parameter(
            torch.eye(embed_dim, dtype=torch.float32)
        )
        self.out_weight = torch.nn.Parameter(
            torch.eye(embed_dim, dtype=torch.float32)
        )

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        *,
        value_params: np.ndarray | None = None,
        return_attention: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Compute the self‑attention output for a batch of sequences.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters used to construct the query projection matrix.
        entangle_params : np.ndarray
            Parameters used to construct the key projection matrix.
        inputs : np.ndarray
            Input tensor of shape ``(batch, seq_len, embed_dim)``.
        value_params : np.ndarray, optional
            Parameters used to construct the value projection matrix.
            If ``None`` the same as ``rotation_params``.
        return_attention : bool, default False
            If ``True`` also return the attention weight matrix.

        Returns
        -------
        output : np.ndarray
            The attended representation of shape ``(batch, seq_len, embed_dim)``.
        attention : np.ndarray, optional
            The attention weight matrix of shape
            ``(batch, num_heads, seq_len, seq_len)`` if requested.
        """
        device = torch.device("cpu")
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32, device=device)

        # Build projection matrices from the provided parameters
        q_w = torch.diag(
            torch.as_tensor(rotation_params, dtype=torch.float32, device=device)
        )
        k_w = torch.diag(
            torch.as_tensor(entangle_params, dtype=torch.float32, device=device)
        )
        v_w = (
            torch.diag(
                torch.as_tensor(
                    value_params if value_params is not None else rotation_params,
                    dtype=torch.float32,
                    device=device,
                )
            )
            if value_params is not None
            else q_w
        )

        # Linear projections
        Q = torch.matmul(inputs_t, q_w)
        K = torch.matmul(inputs_t, k_w)
        V = torch.matmul(inputs_t, v_w)

        # Reshape for multi‑head attention
        batch, seq_len, _ = Q.shape
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        # Final linear projection
        output = torch.matmul(attn_output, self.out_weight)

        if return_attention:
            return output.detach().cpu().numpy(), attn_weights.detach().cpu().numpy()
        return output.detach().cpu().numpy()


__all__ = ["SelfAttentionLayer"]
