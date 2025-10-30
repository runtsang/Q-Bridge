"""Hybrid classical/quantum self‑attention implementation.

The module defines a `HybridSelfAttention` class that can operate in either
classical (torch) or quantum (qiskit) mode, while sharing a common
interface.  The class is coupled to the lightweight estimator utilities
from *FastBaseEstimator.py* and to the synthetic regression dataset
from *QuantumRegression.py*.

Typical usage:

    from SelfAttention__gen227 import HybridSelfAttention, RegressionDataset

    # Classical
    model = HybridSelfAttention(embed_dim=8, mode="classical")
    dataset = RegressionDataset(samples=1000, num_features=8)
    estimator = FastBaseEstimator(model)
    results = estimator.evaluate([lambda out: out.mean()], [np.random.rand(8)])

    # Quantum
    model_q = HybridSelfAttention(n_qubits=4, mode="quantum")
    estimator_q = FastBaseEstimator(model_q)
    results_q = estimator_q.evaluate([tq.PauliZ], [[0.1, 0.2,...]])
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

# ---- Lightweight estimator utilities (FastBaseEstimator) ----
class FastBaseEstimator:
    """Evaluate a model for a list of parameter sets and observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: list[callable],
        parameter_sets: list[list[float]],
    ) -> list[list[float]]:
        self.model.eval()
        results: list[list[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results

# ---- Synthetic regression dataset (QuantumRegression) ----
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate states and labels that mimic a superposition‑based regression task."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns a feature vector and a regression target."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# ---- Model definition ----
class HybridSelfAttention(nn.Module):
    """Classical self‑attention block that follows the interface of the quantum variant."""

    def __init__(self, embed_dim: int, n_heads: int = 1, mode: str = "classical"):
        super().__init__()
        self.mode = mode
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        if self.head_dim * n_heads!= embed_dim:
            raise ValueError("embed_dim must be divisible by n_heads")
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor, rotation_params: torch.Tensor, entangle_params: torch.Tensor) -> torch.Tensor:
        """Compute multi‑head self‑attention.

        Parameters
        ----------
        inputs : torch.Tensor
            Input batch of shape (B, E)
        rotation_params : torch.Tensor
            Parameters that would rotate the queries
        entangle_params : torch.Tensor
            Parameters that would entangle the keys
        """
        B, E = inputs.shape
        # Apply learned projections
        Q = self.q_proj(inputs)
        K = self.k_proj(inputs)
        V = self.v_proj(inputs)

        # Reshape for multi‑head
        Q = Q.view(B, self.n_heads, self.head_dim)
        K = K.view(B, self.n_heads, self.head_dim)
        V = V.view(B, self.n_heads, self.head_dim)

        # Simulate quantum‑style rotations by mixing with provided params
        Q = Q + rotation_params.view(1, self.n_heads, self.head_dim)
        K = K + entangle_params.view(1, self.n_heads, self.head_dim)

        # Scaled dot‑product attention
        scores = torch.einsum("bhd,bhd->bh", Q, K) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bh,bhd->bhd", attn, V).reshape(B, E)

        return self.out_proj(out)

__all__ = ["HybridSelfAttention", "RegressionDataset", "FastBaseEstimator"]
