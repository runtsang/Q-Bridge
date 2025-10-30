"""Utilities for building graph-based quantum neural networks.

This module extends the original seed by adding a variational quantum circuit (VQC)
implemented with Pennylane.  The :class:`GraphQNN` class can be instantiated
with ``backend='vqc'`` to run the quantum model.  The original helper
functions are preserved for compatibility.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import pennylane as qml
import numpy as np
import torch
import torch.nn as nn

QObj = qml.QubitStateVector
Tensor = np.ndarray
Graph = nx.Graph

# --------------------------------------------------------------------------- #
#   Core QNN state propagation – identical to the original seed
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qml.QubitStateVector:
    return qml.Identity(num_qubits)

def _tensored_zero(num_qubits: int) -> qml.QubitStateVector:
    return qml.StateVector(np.zeros(2 ** num_qubits, dtype=complex))

def _swap_registers(op: qml.QubitStateVector, source: int, target: int) -> qml.QubitStateVector:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> qml.QubitStateVector:
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.orth(matrix)
    return qml.QubitStateVector(unitary)

def _random_qubit_state(num_qubits: int) -> qml.QubitStateVector:
    dim = 2 ** num_qubits
    amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amplitudes = amplitudes / np.linalg.norm(amplitudes)
    return qml.QubitStateVector(amplitudes)

def random_training_data(unitary: qml.QubitStateVector, samples: int) -> List[Tuple[qml.QubitStateVector, qml.QubitStateVector]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary @ state))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qml.QubitStateVector]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qml.QubitStateVector] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qml.tensordot(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: qml.QubitStateVector, keep: Sequence[int]) -> qml.QubitStateVector:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qml.QubitStateVector, remove: Sequence[int]) -> qml.QubitStateVector:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qml.QubitStateVector]], layer: int, input_state: qml.QubitStateVector) -> qml.QubitStateVector:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qml.tensordot(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate @ layer_unitary

    return _partial_trace_remove(layer_unitary @ state @ layer_unitary.conj().T, range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qml.QubitStateVector]], samples: Iterable[Tuple[qml.QubitStateVector, qml.QubitStateVector]]):
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qml.QubitStateVector, b: qml.QubitStateVector) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs((a.conj().T @ b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[qml.QubitStateVector], threshold: float, *, secondary: Optional[float] = None, secondary_weight: float = 0.5) -> Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive weight 1.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.
    """
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
#   Variational quantum circuit – minimal Pennylane implementation
# --------------------------------------------------------------------------- #

class _VQC(nn.Module):
    """Variational quantum circuit with one layer of parameterised rotations
    followed by entanglement according to a graph adjacency matrix."""
    def __init__(self, num_qubits: int, num_layers: int = 2):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        # Parameters: (num_layers, num_qubits)
        self.params = nn.Parameter(torch.randn(num_layers, num_qubits))

    def circuit(self, x: np.ndarray, adj: np.ndarray, params: np.ndarray):
        # Encode input features into the initial state via RY rotations
        for q in range(self.num_qubits):
            qml.RY(x[q], wires=q)
        # Variational layers
        for l in range(self.num_layers):
            # Parameterised rotations on each qubit
            for q in range(self.num_qubits):
                qml.RY(params[l, q], wires=q)
            # Entangle according to adjacency
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    if adj[i, j] > 0:
                        qml.CZ(wires=[i, j])
        # Measure expectation of Z on each qubit
        return [qml.expval(qml.PauliZ(q)) for q in range(self.num_qubits)]

class GraphQNN(nn.Module):
    """Unified quantum GraphQNN interface.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture list ``[in_features, hidden1, …, out_features]``.
    backend : str
        Only ``'vqc'`` is supported in this module – a variational quantum circuit.
    """

    def __init__(self, qnn_arch: Sequence[int], backend: str = "vqc"):
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        if backend!= "vqc":
            raise ValueError(f"Unsupported backend {backend!r} for quantum GraphQNN")
        self.backend = backend
        # For simplicity, use the number of input features as the number of qubits
        self.num_qubits = self.qnn_arch[0]
        self.vqc = _VQC(self.num_qubits, num_layers=len(self.qnn_arch) - 1)
        self.dev = qml.device("default.qubit", wires=self.num_qubits)

    def forward(self, features: Tensor, adj: Tensor) -> Tensor:
        """Forward pass through the variational quantum circuit.

        Parameters
        ----------
        features : Tensor
            Node feature vector of shape ``(N,)`` where N is the number of qubits.
        adj : Tensor
            Adjacency matrix of shape ``(N, N)`` used for entanglement.
        """
        @qml.qnode(self.dev, interface="torch")
        def circuit(x, a, params):
            return self.vqc.circuit(x, a, params)

        outputs = circuit(features, adj, self.vqc.params)
        return torch.tensor(outputs, dtype=torch.float32)

    def train_loop(
        self,
        dataset: Iterable[Tuple[Tensor, Tensor]],
        epochs: int = 10,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ) -> None:
        """Simple training loop using Pennylane's gradient descent."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for features, target in dataset:
                optimizer.zero_grad()
                output = self.forward(features, torch.eye(features.size(0), device=device))
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch+1}/{epochs} – loss: {avg_loss:.6f}")

# --------------------------------------------------------------------------- #
#   Exports
# --------------------------------------------------------------------------- #

__all__ = [
    "GraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
