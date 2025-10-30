from __future__ import annotations

import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Tuple

def _clip(value: float, bound: float) -> float:
    """Range‑clamp for parameters, inspired by fraud‑detection clipping."""
    return max(-bound, min(bound, value))

@dataclass
class AttentionParams:
    """Container for all parameters used by the hybrid attention."""
    rotation_params: np.ndarray        # shape (3 * n_qubits,)
    entangle_params: np.ndarray        # shape (n_qubits - 1,)
    fc_thetas: Iterable[float]        # parameters for the FC layer
    clip_bound: float = 5.0            # default clipping bound

class ClassicalSelfAttention(nn.Module):
    """Standard scaled dot‑product attention."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor,
                rot: torch.Tensor,
                ent: torch.Tensor) -> torch.Tensor:
        query = x @ rot
        key   = x @ ent
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return scores

class FullyConnectedLayer(nn.Module):
    """Parameterised linear layer with a tanh activation, mimicking the
    quantum fully‑connected layer in the seed."""
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor, thetas: Iterable[float]) -> torch.Tensor:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation

class SelfAttentionHybrid(nn.Module):
    """Hybrid self‑attention integrating classical and quantum ideas."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim)
        self.fc        = FullyConnectedLayer()
        self.embed_dim = embed_dim

    def forward(self, inputs: torch.Tensor, params: AttentionParams) -> torch.Tensor:
        # Clip parameters before use, following fraud‑detection style
        rot  = torch.as_tensor(
            np.array([_clip(v, params.clip_bound) for v in params.rotation_params]),
            dtype=torch.float32).reshape(self.embed_dim, -1)
        ent  = torch.as_tensor(
            np.array([_clip(v, params.clip_bound) for v in params.entangle_params]),
            dtype=torch.float32).reshape(self.embed_dim, -1)
        # Classical attention weights
        attn = self.attention(inputs, rot, ent)
        # Quantum contribution (if supplied in params)
        if hasattr(params, 'quantum_weights'):
            qw = torch.as_tensor(params.quantum_weights, dtype=torch.float32)
            qw = qw.clamp(-params.clip_bound, params.clip_bound)
            attn = attn * qw.unsqueeze(-1)
        # Value transformation
        fc_out = self.fc(inputs, params.fc_thetas)
        # Weighted sum (broadcast over batch)
        out = torch.sum(attn * fc_out, dim=-1, keepdim=True)
        return out

__all__ = ["SelfAttentionHybrid", "AttentionParams"]
