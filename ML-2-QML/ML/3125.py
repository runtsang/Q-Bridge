"""Hybrid self‑attention module combining classical attention and quantum‑graph propagation.

The module defines `HybridSelfAttentionQNN` which exposes a `run` method that:
  1. Applies a classical self‑attention block.
  2. Encodes the output into a quantum state.
  3. Propagates the state through a sequence of random unitaries.
  4. Builds a fidelity‑based adjacency graph from the intermediate states.

The implementation uses NumPy/PyTorch for the classical part and torch for the quantum‑like state manipulation, so it can be executed on a CPU without a real quantum backend.
"""

from __future__ import annotations

import itertools
import numpy as np
import torch
import networkx as nx
from typing import Iterable, List, Tuple

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Classical self‑attention
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention:
    """Simple self‑attention block that mirrors the interface of the
    quantum implementation.  It accepts rotation and entanglement
    matrices and applies them to the input features.
    """
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        # Reshape parameters to match the embedding dimension
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# --------------------------------------------------------------------------- #
# Quantum‑like state utilities
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> torch.Tensor:
    """Return a 2ⁿ×2ⁿ identity tensor."""
    dim = 2**num_qubits
    return torch.eye(dim, dtype=torch.complex64)

def _tensored_zero(num_qubits: int) -> torch.Tensor:
    """Return a |0…0⟩ column vector."""
    dim = 2**num_qubits
    state = torch.zeros((dim, 1), dtype=torch.complex64)
    state[0, 0] = 1.0
    return state

def _random_unitary(num_qubits: int) -> torch.Tensor:
    """Generate a random Haar‑distributed unitary on 2ⁿ dimensions."""
    dim = 2**num_qubits
    mat = torch.randn((dim, dim), dtype=torch.complex64)
    mat += 1j * torch.randn((dim, dim), dtype=torch.complex64)
    q, _ = torch.linalg.qr(mat)
    return q

def _random_state(num_qubits: int) -> torch.Tensor:
    """Sample a random pure state on 2ⁿ qubits."""
    dim = 2**num_qubits
    vec = torch.randn((dim, 1), dtype=torch.complex64)
    vec += 1j * torch.randn((dim, 1), dtype=torch.complex64)
    return vec / torch.norm(vec)

# --------------------------------------------------------------------------- #
# Layer propagation
# --------------------------------------------------------------------------- #
def _layer_channel(
    n_inputs: int,
    n_outputs: int,
    unitary: torch.Tensor,
    input_state: torch.Tensor,
) -> torch.Tensor:
    """
    Concatenate the input state with a |0…0⟩ ancilla of size n_outputs,
    apply the unitary, and trace out the ancilla qubits.
    """
    ancilla = _tensored_zero(n_outputs)
    state = torch.kron(input_state, ancilla)
    out = unitary @ state
    # Partial trace: keep the first n_inputs qubits
    dim_total = 2 ** (n_inputs + n_outputs)
    out = out.reshape([2]* (n_inputs + n_outputs))
    # Move the qubits to be kept to the front
    order = list(range(n_inputs)) + list(range(n_inputs, n_inputs + n_outputs))
    out = out.permute(*order)
    out = out.reshape([2**n_inputs, 1])
    return out

def _state_fidelity(state_a: torch.Tensor, state_b: torch.Tensor) -> float:
    """Squared overlap between two pure states."""
    return float((state_a.conj().T @ state_b)[0, 0].abs().item()**2)

def fidelity_graph(states: List[torch.Tensor], threshold: float, *, secondary: float | None = None) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j in range(i+1, len(states)):
            fid = _state_fidelity(a, states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=0.5)
    return graph

# --------------------------------------------------------------------------- #
# Hybrid model
# --------------------------------------------------------------------------- #
class HybridSelfAttentionQNN:
    """
    Hybrid model that first applies classical self‑attention, then
    propagates the resulting vector through a random‑unitary quantum
    neural network and finally returns a fidelity graph.
    """
    def __init__(
        self,
        embed_dim: int,
        n_qubits: int,
        qnn_arch: List[int],
    ):
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.qnn_arch = qnn_arch
        self.attention = ClassicalSelfAttention(embed_dim)
        # Pre‑generate the random unitaries for each layer
        self.unitaries: List[torch.Tensor] = []
        for layer in range(1, len(qnn_arch)):
            n_in = qnn_arch[layer-1]
            n_out = qnn_arch[layer]
            # unitary acts on n_in+1 qubits (input + one ancilla)
            self.unitaries.append(_random_unitary(n_in+1))

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        fidelity_threshold: float = 0.8,
        *,
        secondary_threshold: float | None = None,
    ) -> dict:
        """
        Execute the hybrid pipeline.

        Parameters
        ----------
        inputs : np.ndarray
            Feature matrix of shape (batch, embed_dim).
        rotation_params, entangle_params : np.ndarray
            Parameters for the classical self‑attention block.
        fidelity_threshold : float
            Threshold for edges in the fidelity graph.
        secondary_threshold : float | None
            Optional secondary threshold for weighted edges.

        Returns
        -------
        dict
            ``{'attention': np.ndarray,
                 'quantum_states': List[np.ndarray],
                 'fidelity_graph': nx.Graph}``
        """
        # 1. Classical self‑attention
        attn_out = self.attention.run(rotation_params, entangle_params, inputs)

        # 2. Encode the attention output into a quantum state
        #   We take the first sample for simplicity; extendable to batch mode.
        vec = torch.as_tensor(attn_out[0], dtype=torch.complex64)
        # Normalize
        vec = vec / torch.norm(vec)
        # If the vector length does not match 2ⁿ, pad with zeros
        dim = 2**self.n_qubits
        if vec.shape[0] < dim:
            padded = torch.zeros((dim, 1), dtype=torch.complex64)
            padded[:vec.shape[0], 0] = vec
            vec = padded
        elif vec.shape[0] > dim:
            vec = vec[:dim, 0].reshape((dim, 1))
        else:
            vec = vec.reshape((dim, 1))

        # 3. Propagate through the quantum layers
        states: List[torch.Tensor] = [vec]
        current = vec
        for layer, unitary in enumerate(self.unitaries, start=1):
            n_in = self.qnn_arch[layer-1]
            n_out = self.qnn_arch[layer]
            current = _layer_channel(n_in, n_out, unitary, current)
            states.append(current)

        # 4. Build fidelity graph
        graph = fidelity_graph(
            [s.numpy() for s in states],
            threshold=fidelity_threshold,
            secondary=secondary_threshold,
        )

        return {
            "attention": attn_out,
            "quantum_states": [s.numpy() for s in states],
            "fidelity_graph": graph,
        }

__all__ = ["HybridSelfAttentionQNN"]
