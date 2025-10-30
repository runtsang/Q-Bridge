"""Importable classical module defining UnifiedSelfAttention.

This module integrates classical self‑attention, a
fully‑connected layer, a lightweight estimator, and an RBF kernel.
"""

import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence, Callable, Any

# ----- Classical primitives -----------------------------------------
class _ClassicalAttention(nn.Module):
    """Multi‑head self‑attention implemented with torch linear layers."""
    def __init__(self, embed_dim: int, heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_weights = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1
        )
        attn_output = torch.matmul(attn_weights, v)
        return self.out_proj(attn_output)

class _FullyConnectedLayer(nn.Module):
    """Simple linear layer used as a stand‑in for a quantum FCL."""
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))

class _FastEstimator:
    """Estimator that evaluates a model on many parameter sets."""
    def __init__(self, model: nn.Module, shots: int | None = None, seed: int | None = None):
        self.model = model
        self.shots = shots
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def evaluate(self, observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]]) -> list[list[float]]:
        self.model.eval()
        results: list[list[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).view(-1, 1)
                outputs = self.model(inputs)
                row: list[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    row.append(float(val))
                results.append(row)
        if self.shots is not None:
            noisy = []
            for row in results:
                noisy.append([float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row])
            return noisy
        return results

class _Kernel:
    """Radial Basis Function kernel implemented with torch."""
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[torch.exp(-self.gamma * torch.sum((x - y) ** 2)).item()
                          for y in b] for x in a])

# ----- Unified module -----------------------------------------------
class UnifiedSelfAttention:
    """Hybrid self‑attention module that can operate in classical or quantum mode."""
    def __init__(self,
                 embed_dim: int = 4,
                 heads: int = 1,
                 n_qubits: int = 4,
                 use_quantum: bool = False,
                 backend: Any | None = None,
                 seed: int | None = None):
        self.embed_dim = embed_dim
        self.heads = heads
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum
        self.backend = backend
        self.seed = seed

        # Classical sub‑modules
        self.classical_attention = _ClassicalAttention(embed_dim, heads)
        self.fcl = _FullyConnectedLayer(embed_dim)
        self.estimator = _FastEstimator(self.fcl, shots=1000, seed=seed)
        self.kernel = _Kernel(gamma=1.0)

    # ------------------------------------------------------------------
    def set_parameters(self,
                       rotation_params: np.ndarray | None = None,
                       entangle_params: np.ndarray | None = None) -> None:
        """Store parameters for the quantum part (ignored in classical mode)."""
        self.rotation_params = rotation_params
        self.entangle_params = entangle_params

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the attention output, either classical or a quantum kernel."""
        if self.use_quantum:
            # For the classical module we can still return a kernel similarity
            return torch.tensor(self.kernel.matrix([x], [x])[0, 0])
        return self.classical_attention(x)

    # ------------------------------------------------------------------
    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            shots: int = 1024,
            backend: Any | None = None) -> np.ndarray:
        """Hybrid run method mimicking the original SelfAttention API."""
        self.set_parameters(rotation_params, entangle_params)
        inp_tensor = torch.as_tensor(inputs, dtype=torch.float32)
        if self.use_quantum:
            # In quantum mode we evaluate a variational circuit that
            # approximates a self‑attention weight matrix.
            # Here we use a simple linear transform as a placeholder.
            weight = self.rotation_params.reshape(-1, inp_tensor.shape[-1])
            attn = torch.matmul(inp_tensor, weight)
            output = self.fcl(attn)
        else:
            output = self.classical_attention(inp_tensor)
        return output.detach().numpy()

    # ------------------------------------------------------------------
    def estimate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]]) -> list[list[float]]:
        """Evaluate a list of observables over many parameter sets."""
        return self.estimator.evaluate(observables, parameter_sets)

    # ------------------------------------------------------------------
    def kernel_matrix(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return a Gram matrix using the embedded RBF kernel."""
        return self.kernel.matrix(a, b)
