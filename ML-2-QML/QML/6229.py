"""Hybrid quantum convolutional filter and graph neural network.

This module defines ConvGraphQNN, a quantum network that replaces the classical
convolution with a variational quanvolution circuit and then applies a graph‑based
message passing layer that uses fidelity of the resulting probability vectors to build the
adjacency.  The implementation uses Qiskit and PyTorch for hybrid execution.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import networkx as nx
import qiskit
import torch

Tensor = torch.Tensor

def _random_circuit(num_qubits: int, depth: int = 2) -> qiskit.QuantumCircuit:
    """Generate a random parameterised circuit (no measurement)."""
    circuit = qiskit.QuantumCircuit(num_qubits)
    for _ in range(depth):
        circuit.barrier()
        for q in range(num_qubits):
            theta = qiskit.circuit.Parameter(f"theta{q}")
            circuit.rx(theta, q)
        circuit += qiskit.circuit.random.random_circuit(num_qubits, depth, seed=42)
    return circuit

def random_training_data(unitary: qiskit.QuantumCircuit, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate random training pairs (state, target_state)."""
    data = []
    for _ in range(samples):
        init_state = np.zeros(2 ** unitary.num_qubits, dtype=complex)
        init_state[0] = 1.0
        unitary_matrix = qiskit.quantum_info.Operator(unitary).data
        state = unitary_matrix @ init_state
        target_state = unitary_matrix @ state
        data.append((torch.from_numpy(state), torch.from_numpy(target_state)))
    return data

def random_network(qnn_arch: Sequence[int], samples: int):
    """Construct a random quantum network of the given architecture."""
    circuits: List[List[qiskit.QuantumCircuit]] = [[]]
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        layer_ops: List[qiskit.QuantumCircuit] = []
        for _ in range(out_f):
            op = _random_circuit(in_f + 1)
            layer_ops.append(op)
        circuits.append(layer_ops)
    target_circuit = _random_circuit(qnn_arch[-1])
    training_data = random_training_data(target_circuit, samples)
    return list(qnn_arch), circuits, training_data, target_circuit

def feedforward(
    qnn_arch: Sequence[int],
    circuits: Sequence[Sequence[qiskit.QuantumCircuit]],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Propagate quantum states through the network."""
    stored: List[List[Tensor]] = []
    for state, _ in samples:
        layerwise: List[Tensor] = [state]
        current_state = state
        for layer in range(1, len(qnn_arch)):
            for op in circuits[layer]:
                op_matrix = qiskit.quantum_info.Operator(op).data
                current_state = torch.from_numpy(op_matrix @ current_state.numpy())
            layerwise.append(current_state)
        stored.append(layerwise)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap between two pure state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class QuanvCircuit:
    """Variational quanvolution circuit that encodes a 2‑D kernel into qubit rotations."""

    def __init__(
        self,
        kernel_size: int,
        backend: qiskit.providers.Backend,
        shots: int,
        threshold: float,
    ):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [
            qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)
        ]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2, seed=42)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: torch.Tensor) -> np.ndarray:
        """Execute the quantum circuit on a 2‑D kernel and return a probability vector."""
        data_flat = data.flatten().numpy()
        param_bind = {self.theta[i]: math.pi if val > self.threshold else 0.0 for i, val in enumerate(data_flat)}
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result().get_counts(self._circuit)
        probs = np.zeros(self.n_qubits)
        for bitstring, count in result.items():
            bits = np.array([int(b) for b in bitstring[::-1]])  # little‑endian
            probs += bits * count
        probs /= self.shots
        return probs

class ConvGraphQNN:
    """Hybrid quantum convolution + graph neural network.

    The network applies a variational quanvolution to each kernel, then builds
    a graph of samples using fidelity of the resulting probability vectors.
    A single message‑passing step averages neighbour vectors to generate the
    updated outputs.  The implementation is a drop‑in replacement for the
    classical ConvGraphQNN and is fully compatible with PyTorch.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.Backend | None = None,
        shots: int = 1024,
        conv_threshold: float = 0.5,
        graph_threshold: float = 0.8,
        secondary_threshold: float | None = None,
        secondary_weight: float = 0.5,
    ):
        self.kernel_size = kernel_size
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.conv_threshold = conv_threshold
        self.graph_threshold = graph_threshold
        self.secondary_threshold = secondary_threshold
        self.secondary_weight = secondary_weight
        self.quantum_filter = QuanvCircuit(
            kernel_size=kernel_size,
            backend=self.backend,
            shots=shots,
            threshold=conv_threshold,
        )

    def forward(self, data: Sequence[torch.Tensor]) -> List[np.ndarray]:
        """Process a batch of 2‑D kernels.

        Args:
            data: Sequence of tensors each of shape (kernel_size, kernel_size).

        Returns:
            List of updated probability vectors after graph propagation.
        """
        prob_vectors = [self.quantum_filter.run(kernel) for kernel in data]

        graph = fidelity_adjacency(
            prob_vectors,
            self.graph_threshold,
            secondary=self.secondary_threshold,
            secondary_weight=self.secondary_weight,
        )

        updated = []
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if neighbors:
                neighbor_vals = np.stack([prob_vectors[n] for n in neighbors])
                updated_val = np.mean(neighbor_vals, axis=0)
            else:
                updated_val = prob_vectors[node]
            updated.append(updated_val)

        return updated

__all__ = [
    "ConvGraphQNN",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "QuanvCircuit",
]
