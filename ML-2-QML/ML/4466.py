"""Hybrid classical module that unifies self‑attention, kernel, and graph QNN.

The module exposes a single ``HybridSelfAttentionKernelQNN`` class that
provides three mutually compatible sub‑modules:

* ``attention`` – a differentiable self‑attention layer (torch‑based)
* ``kernel``    – an RBF kernel or a simulated quantum kernel
* ``qnn``       – a graph‑based neural network that can be run
  either classically or via a simple quantum simulation.

The class can be instantiated with ``use_quantum=True`` to enable the
quantum‑style approximations; otherwise it falls back to purely
classical implementations.  The API mirrors the original
``SelfAttention`` seed while adding the new kernel and graph
capabilities.

The design keeps the file lightweight so that downstream projects
can import the class directly from
``seed_codebase/ML-Github/SelfAttention__gen146.py`` and experiment
with the hybrid workflow.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Iterable, List, Sequence, Tuple


# --------------------------------------------------------------------------- #
#  Classical self‑attention helper
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """Differentiable self‑attention block that mirrors the quantum interface."""

    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        # The rotation_params/entangle_params are accepted for API
        # compatibility but ignored in the classical implementation.
        query = self.query_proj(x)
        key   = self.key_proj(x)
        value = self.value_proj(x)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value


# --------------------------------------------------------------------------- #
#  Quantum‑style self‑attention (classical simulation)
# --------------------------------------------------------------------------- #
class QuantumSelfAttention(nn.Module):
    """A lightweight classical simulation of a quantum self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimension of the embedding space (default 4).  The unitary
        matrix is constructed from ``rotation_params`` and
        ``entangle_params`` and applied to the input state vector.
    """

    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def _build_unitary(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        # Build a simple unitary matrix from the parameters.
        # For each qubit we create a 2×2 rotation matrix and take the tensor product.
        U = np.eye(2)
        for i in range(self.embed_dim // 2):
            a, b, c = rotation_params[3 * i : 3 * (i + 1)]
            Rx = np.array([[np.cos(a / 2), -1j * np.sin(a / 2)],
                           [-1j * np.sin(a / 2), np.cos(a / 2)]])
            Ry = np.array([[np.cos(b / 2), -np.sin(b / 2)],
                           [np.sin(b / 2), np.cos(b / 2)]])
            Rz = np.array([[np.exp(-1j * c / 2), 0],
                           [0, np.exp(1j * c / 2)]])
            U = np.kron(U, Rx @ Ry @ Rz)
        # Entanglement via simple CNOT‑like mixing
        for i in range(self.embed_dim - 1):
            theta = entangle_params[i]
            CNOT = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, np.cos(theta), -np.sin(theta)],
                             [0, 0, np.sin(theta),  np.cos(theta)]])
            U = CNOT @ U
        return U

    def forward(
        self,
        x: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        U = self._build_unitary(rotation_params, entangle_params)
        # Apply unitary to each sample in the batch
        batch = x.shape[0]
        x_vec = x.reshape(batch, self.embed_dim)
        # Convert to numpy for the simulation
        x_np = x_vec.detach().cpu().numpy()
        y_np = x_np @ U.T
        y = torch.from_numpy(y_np).to(x.device).float()
        # Treat the output as a new embedding and compute attention
        query = y
        key   = y
        value = y
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value


# --------------------------------------------------------------------------- #
#  Classical RBF kernel
# --------------------------------------------------------------------------- #
class ClassicalKernel(nn.Module):
    """Radial‑basis‑function kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


# --------------------------------------------------------------------------- #
#  Quantum‑style kernel (classical simulation)
# --------------------------------------------------------------------------- #
class QuantumKernel(nn.Module):
    """Simulated quantum kernel that uses an inner‑product of encoded
    state vectors.  The encoding is a simple amplitude encoding.
    """

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Fixed unitary for simulation (identity)
        self.U = np.eye(2 ** self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Simple amplitude encoding: normalize the vector
        x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-12)
        y_norm = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-12)
        # Apply fixed unitary
        x_enc = torch.from_numpy(self.U @ x_norm.T).T
        y_enc = torch.from_numpy(self.U @ y_norm.T).T
        # Inner product squared
        return torch.abs(torch.sum(x_enc * y_enc, dim=-1, keepdim=True)) ** 2


# --------------------------------------------------------------------------- #
#  Classical graph‑based neural network
# --------------------------------------------------------------------------- #
class ClassicalGraphQNN(nn.Module):
    """Graph‑structured feed‑forward network using tanh activations."""

    def __init__(self, qnn_arch: Sequence[int]) -> None:
        super().__init__()
        self.arch = list(qnn_arch)
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(out, in_)) for in_, out in zip(self.arch[:-1], self.arch[1:])]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        activations: List[torch.Tensor] = [x]
        current = x
        for weight in self.weights:
            current = torch.tanh(current @ weight.T)
            activations.append(current)
        return activations


# --------------------------------------------------------------------------- #
#  Quantum‑style graph QNN (classical simulation)
# --------------------------------------------------------------------------- #
class QuantumGraphQNN(nn.Module):
    """Simulated quantum graph neural network that operates on pure states.

    The network is a sequence of random unitary layers applied to the
    input state.  The output is the probability distribution over the
    computational basis after measurement.
    """

    def __init__(self, qnn_arch: Sequence[int]) -> None:
        super().__init__()
        self.arch = list(qnn_arch)
        self.num_qubits = self.arch[-1]
        # Random unitary per layer
        self.unitaries = [self._random_unitary(self.num_qubits) for _ in self.arch[1:]]

    def _random_unitary(self, dim: int) -> np.ndarray:
        mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        q, _ = np.linalg.qr(mat)
        return q

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Encode input as a state vector
        batch = x.shape[0]
        state = x.reshape(batch, -1)
        # Normalize
        state = state / (torch.norm(state, dim=-1, keepdim=True) + 1e-12)
        activations: List[torch.Tensor] = [state]
        current = state
        for U in self.unitaries:
            # Apply unitary
            U_torch = torch.from_numpy(U).to(x.device).float()
            current = current @ U_torch.T
            # Measurement probabilities (squared amplitudes)
            probs = torch.abs(current) ** 2
            activations.append(probs)
        return activations


# --------------------------------------------------------------------------- #
#  Hybrid controller
# --------------------------------------------------------------------------- #
class HybridSelfAttentionKernelQNN(nn.Module):
    """Hybrid module that combines self‑attention, kernel, and graph QNN."""

    def __init__(
        self,
        embed_dim: int = 4,
        qnn_arch: Sequence[int] = (4, 8, 4),
        gamma: float = 1.0,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.attention = (
            QuantumSelfAttention(embed_dim) if use_quantum else ClassicalSelfAttention(embed_dim)
        )
        self.kernel = QuantumKernel() if use_quantum else ClassicalKernel(gamma)
        self.qnn = QuantumGraphQNN(qnn_arch) if use_quantum else ClassicalGraphQNN(qnn_arch)

    def forward(
        self,
        x: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Return a tuple of (attention_output, kernel_output, qnn_output)."""
        attn_out = self.attention(x, rotation_params, entangle_params)
        ker_out = torch.stack([self.kernel(x[i], x[i]) for i in range(x.shape[0])])
        qnn_out = self.qnn(x)
        return attn_out, ker_out, qnn_out


def SelfAttention(use_quantum: bool = False, **kwargs) -> HybridSelfAttentionKernelQNN:
    """Convenient factory that mirrors the original SelfAttention interface."""
    return HybridSelfAttentionKernelQNN(use_quantum=use_quantum, **kwargs)


__all__ = [
    "HybridSelfAttentionKernelQNN",
    "SelfAttention",
]
