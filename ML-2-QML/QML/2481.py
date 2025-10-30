"""Quantum‑enhanced SelfAttention with graph‑based fidelity layer.

This module merges the quantum self‑attention circuit from the original
SelfAttention seed with the graph utilities from GraphQNN.  The class
SelfAttention builds a variational circuit that first encodes the input
vector via Rx/Ry/Rz gates, then applies a series of entangling CRX
gates.  After execution on a backend it returns measurement counts or
statevectors.  The class also offers a method to construct a weighted
graph from the fidelities of the resulting states, enabling hybrid
training.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Iterable as IterableType

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector

Tensor = np.ndarray
Array = np.ndarray

def _random_unitary(num_qubits: int) -> QuantumCircuit:
    """Generate a random unitary circuit on ``num_qubits`` qubits."""
    qc = QuantumCircuit(num_qubits)
    for _ in range(3 * num_qubits):
        qc.ry(np.random.uniform(0, 2 * np.pi), np.random.randint(num_qubits))
        qc.rz(np.random.uniform(0, 2 * np.pi), np.random.randint(num_qubits))
    return qc

def _random_training_data(unitary: QuantumCircuit, samples: int) -> List[Tuple[QuantumCircuit, QuantumCircuit]]:
    """Generate training pairs (input_circuit, target_circuit) using a random unitary."""
    dataset: List[Tuple[QuantumCircuit, QuantumCircuit]] = []
    for _ in range(samples):
        input_circ = _random_unitary(unitary.num_qubits)
        target_circ = unitary.compose(input_circ)
        dataset.append((input_circ, target_circ))
    return dataset

class SelfAttention:
    """Hybrid quantum self‑attention with graph‑based fidelity utilities."""

    def __init__(self, n_qubits: int = 4, qnn_arch: Sequence[int] | None = None):
        self.n_qubits = n_qubits
        self.qnn_arch = list(qnn_arch) if qnn_arch else [n_qubits, n_qubits, n_qubits]
        self.backend = Aer.get_backend("statevector_simulator")

        # Quantum self‑attention circuit parameters
        self.rotation_params = np.random.uniform(0, 2 * np.pi, size=(n_qubits, 3))
        self.entangle_params = np.random.uniform(0, 2 * np.pi, size=(n_qubits - 1,))

        # Quantum‑neural‑network data
        self._qnn_circuits: List[List[QuantumCircuit]] | None = None
        self._qnn_training_data: List[Tuple[QuantumCircuit, QuantumCircuit]] | None = None

    # ------------------------------------------------------------------ #
    #  Quantum self‑attention circuit
    # ------------------------------------------------------------------ #
    def _build_circuit(self, rotation_params: Array, entangle_params: Array) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[i, 0], i)
            qc.ry(rotation_params[i, 1], i)
            qc.rz(rotation_params[i, 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure_all()
        return qc

    def run(self, backend=None, rotation_params: Array | None = None,
            entangle_params: Array | None = None, shots: int = 1024) -> dict:
        """Execute the self‑attention circuit and return measurement counts."""
        if backend is None:
            backend = self.backend
        if rotation_params is None:
            rotation_params = self.rotation_params
        if entangle_params is None:
            entangle_params = self.entangle_params
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

    # ------------------------------------------------------------------ #
    #  Quantum‑graph construction
    # ------------------------------------------------------------------ #
    def build_qnn(self, qnn_arch: Sequence[int] | None = None) -> None:
        """Generate a random quantum‑neural‑network and its training data."""
        if qnn_arch is None:
            qnn_arch = self.qnn_arch
        self._qnn_circuits = []
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops = []
            for _ in range(num_outputs):
                layer_ops.append(_random_unitary(num_inputs + 1))
            self._qnn_circuits.append(layer_ops)

        # training data: use the last layer unitary as target
        target = self._qnn_circuits[-1][0]
        self._qnn_training_data = _random_training_data(target, samples=100)

    def feedforward(self, samples: IterableType[Tuple[QuantumCircuit, QuantumCircuit]]) -> List[List[QuantumCircuit]]:
        """Apply the quantum‑feedforward to a set of input states."""
        if self._qnn_circuits is None:
            raise ValueError("call ``build_qnn()`` before feeding forward")
        activations: List[List[QuantumCircuit]] = []
        for input_circ, _ in samples:
            layerwise = [input_circ]
            current = input_circ
            for ops in self._qnn_circuits:
                state = current
                for gate in ops:
                    state = gate.compose(state)
                layerwise.append(state)
            activations.append(layerwise)
        return activations

    # ------------------------------------------------------------------ #
    #  Fidelity‑based graph construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _statevector(circ: QuantumCircuit) -> Statevector:
        """Return the statevector of a circuit on the statevector simulator."""
        result = execute(circ, Aer.get_backend("statevector_simulator")).result()
        return Statevector(result.get_statevector(circ))

    @staticmethod
    def state_fidelity(a: QuantumCircuit, b: QuantumCircuit) -> float:
        """Compute the fidelity between two statevectors produced by the circuits."""
        sv_a = SelfAttention._statevector(a)
        sv_b = SelfAttention._statevector(b)
        return float(sv_a.fidelity(sv_b))

    def fidelity_graph(self, threshold: float, *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        """Create a weighted graph from the fidelities of the last‑layer states."""
        if self._qnn_circuits is None:
            raise ValueError("call ``build_qnn()`` before constructing graph")
        last_states = [target for _, target in self._qnn_training_data]
        graph = nx.Graph()
        graph.add_nodes_from(range(len(last_states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(last_states), 2):
            fid = self.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------ #
    #  Graph utilities from GraphQNN
    # ------------------------------------------------------------------ #
    @staticmethod
    def fidelity_adjacency(states: Sequence[QuantumCircuit], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Create a weighted adjacency graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = SelfAttention.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = ["SelfAttention"]
