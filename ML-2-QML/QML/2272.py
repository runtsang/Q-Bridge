"""Quantum graph QNN utilities.

This module implements a lightweight quantum neural network based on
the GraphQNN seed.  It provides a single public function
`run_quantum_graph` that accepts a classical vector, encodes it
into a quantum state, propagates it through a random unitary network,
computes fidelities between the layer states, and returns a
NetworkX graph weighted by those fidelities.

The implementation uses Qutip for state manipulation and
NetworkX for graph construction.  The function is fully
self‑contained and can be called from the classical module.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Sequence as Seq

import networkx as nx
import numpy as np
import qutip as qt

# --------------------------------------------------------------------------- #
#  Utility functions
# --------------------------------------------------------------------------- #
def _random_unitary(num_qubits: int) -> qt.Qobj:
    """Generate a random unitary matrix for `num_qubits` qubits."""
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    # QR decomposition to get a unitary
    q, _ = np.linalg.qr(mat)
    return qt.Qobj(q, dims=[[2] * num_qubits, [2] * num_qubits])

def _random_state(num_qubits: int) -> qt.Qobj:
    """Generate a random pure state for `num_qubits` qubits."""
    vec = np.random.randn(2 ** num_qubits) + 1j * np.random.randn(2 ** num_qubits)
    vec /= np.linalg.norm(vec)
    return qt.Qobj(vec, dims=[[2] * num_qubits, [1] * num_qubits])

# --------------------------------------------------------------------------- #
#  Quantum graph QNN
# --------------------------------------------------------------------------- #
def _build_unitary_layers(qnn_arch: Sequence[int]) -> list[list[qt.Qobj]]:
    """Build a list of unitary layers mirroring the GraphQNN seed."""
    layers: list[list[qt.Qobj]] = [[]]
    for layer_idx in range(1, len(qnn_arch)):
        in_f = qnn_arch[layer_idx - 1]
        out_f = qnn_arch[layer_idx]
        ops: list[qt.Qobj] = []
        for out in range(out_f):
            # Each output node gets a distinct unitary gate that acts on
            # the concatenated input and an auxiliary qubit.
            ops.append(_random_unitary(in_f + 1))
        layers.append(ops)
    return layers

def _layer_output(
    layer: int,
    state: qt.Qobj,
    qnn_arch: Sequence[int],
    layers: Sequence[list[qt.Qobj]],
) -> qt.Qobj:
    """Apply the unitary for a single layer and return the partial trace."""
    num_in = qnn_arch[layer - 1]
    out_f = qnn_arch[layer]
    # Append auxiliary zero state for the output qubits
    state = qt.tensor(state, qt.qeye(2 ** out_f))
    # Build unitary
    unitary = layers[layer][0]
    for gate in layers[layer][1:]:
        unitary = gate * unitary
    # Apply unitary
    state = unitary * state * unitary.dag()
    # Keep output qubits
    keep = list(range(num_in, num_in + out_f))
    return qt.ptrace(state, keep)

def _propagate(
    data: np.ndarray,
    qnn_arch: Sequence[int],
    layers: Sequence[list[qt.Qobj]],
) -> list[qt.Qobj]:
    """Propagate a classical vector through the QNN and return all layer states."""
    # Encode data as a pure state via amplitude encoding.
    # If data length is not a power of two, pad with zeros.
    target_len = 2 ** (len(qnn_arch) - 1)
    if data.size < target_len:
        data = np.pad(data, (0, target_len - data.size), "constant")
    else:
        data = data[:target_len]
    state = qt.Qobj(data, dims=[[2] * (len(qnn_arch) - 1), [1] * (len(qnn_arch) - 1)])
    # Normalize
    state = state / state.norm()
    states = [state]
    current = state
    for layer in range(1, len(qnn_arch)):
        current = _layer_output(layer, current, qnn_arch, layers)
        states.append(current)
    return states

# --------------------------------------------------------------------------- #
#  Fidelity and graph construction
# --------------------------------------------------------------------------- #
def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the fidelity between two (mixed) quantum states."""
    return qt.fidelity(a, b)

def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def run_quantum_graph(
    data: np.ndarray,
    qnn_arch: Sequence[int],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """
    Run a quantum graph QNN on the provided data and return a
    fidelity‑based adjacency graph.

    Parameters
    ----------
    data : np.ndarray
        1‑D array of real numbers that will be encoded as a quantum
        state.  The length is padded/truncated to the nearest power of
        two that matches the QNN architecture.
    qnn_arch : Sequence[int]
        Architecture of the QNN: a list of layer widths.
    threshold : float
        Fidelity threshold for creating edges with weight 1.
    secondary : float | None, optional
        Lower fidelity threshold for adding edges with a smaller
        weight.  If ``None`` no secondary edges are added.
    secondary_weight : float, optional
        Weight assigned to secondary edges.

    Returns
    -------
    nx.Graph
        Weighted graph where nodes correspond to the layer states
        and edges represent fidelities above the specified thresholds.
    """
    layers = _build_unitary_layers(qnn_arch)
    states = _propagate(data, qnn_arch, layers)
    return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = [
    "run_quantum_graph",
    "state_fidelity",
    "fidelity_adjacency",
]
