"""HybridAttentionLayer – classical implementation.

The class combines a simple linear transform with a self‑attention
mechanism.  It mimics the interface of the original FCL and SelfAttention
modules, allowing seamless swapping between classical and quantum
back‑ends.

Typical usage:

    from HybridAttentionLayer import HybridAttentionLayer
    layer = HybridAttentionLayer(n_features=3, n_qubits=4)
    output = layer.run(
        thetas=[0.1, 0.2, 0.3,  # linear weight
                0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12,  # rotations
                0.13, 0.14, 0.15, 0.16],  # entanglements
        inputs=[[1.0, 0.5, -0.2]]
    )
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ClassicalSelfAttention:
    """Pure‑Python self‑attention block used by the hybrid layer."""

    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        # Linear projections
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)

        # Attention scores
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)

        # Weighted sum of values
        return (scores @ value).numpy()


class HybridAttentionLayer(nn.Module):
    """Hybrid classical layer combining linear and self‑attention."""

    def __init__(self,
                 n_features: int = 1,
                 n_qubits: int = 4,
                 embed_dim: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        self.n_features = n_features
        self.n_qubits = n_qubits

    def run(self, thetas: list[float], inputs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        thetas : list[float]
            Concatenated parameters:
            - first ``n_features`` values for the linear weight
            - next ``3 * n_qubits`` values for rotation angles
            - remaining ``n_qubits - 1`` values for entanglement angles
        inputs : np.ndarray
            Batch of input vectors, shape ``(batch, n_features)``.

        Returns
        -------
        np.ndarray
            Combined output of the linear and attention blocks.
        """
        # Split parameters
        lin_params = thetas[: self.n_features]
        rot_params = thetas[self.n_features:
                            self.n_features + 3 * self.n_qubits]
        ent_params = thetas[self.n_features + 3 * self.n_qubits :]

        # Linear layer
        self.linear.weight.data = torch.tensor(
            [lin_params], dtype=torch.float32
        )
        self.linear.bias.data = torch.tensor([0.0], dtype=torch.float32)
        lin_out = self.linear(torch.as_tensor(inputs, dtype=torch.float32)).detach().numpy()

        # Self‑attention
        attn_out = self.attention.run(rot_params, ent_params, inputs)

        # Simple averaging of the two contributions
        return (lin_out + attn_out) / 2.0


__all__ = ["HybridAttentionLayer"]
