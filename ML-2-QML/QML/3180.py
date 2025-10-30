"""Quantum hybrid self‑attention + graph neural network.

The class HybridSelfAttentionGraphQNN implements a variational self‑attention
circuit that can be executed on a simulator or a real device.  It also
provides utilities for generating random training data, propagating states
through a layered unitary network, and building a fidelity‑based graph
from the resulting states.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator

__all__ = ["HybridSelfAttentionGraphQNN"]


class HybridSelfAttentionGraphQNN:
    """Variational self‑attention circuit with graph utilities."""

    def __init__(self, n_qubits: int = 4, embed_dim: int = 4):
        self.n_qubits = n_qubits
        self.embed_dim = embed_dim
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        """Construct a parameterised self‑attention circuit."""
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], qr[i])
            circuit.ry(rotation_params[3 * i + 1], qr[i])
            circuit.rz(rotation_params[3 * i + 2], qr[i])
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], qr[i], qr[i + 1])
        circuit.measure(qr, cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """Execute the circuit and return measurement counts."""
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

    @staticmethod
    def random_training_data(
        unitary: Operator, samples: int
    ) -> List[Tuple[Statevector, Statevector]]:
        """Generate random input–target pairs for a target unitary."""
        data: List[Tuple[Statevector, Statevector]] = []
        for _ in range(samples):
            state = Statevector.random(unitary.num_qubits)
            data.append((state, unitary @ state))
        return data

    def random_network(
        self, arch: List[int], samples: int
    ) -> Tuple[List[int], List[Operator], List[Tuple[Statevector, Statevector]], Operator]:
        """Create a random layered unitary network and training data."""
        target_unitary = Operator(qiskit.quantum_info.random_unitary(2 ** self.n_qubits))
        training_data = self.random_training_data(target_unitary, samples)
        unitaries: List[Operator] = [
            Operator(qiskit.quantum_info.random_unitary(2 ** self.n_qubits))
            for _ in range(len(arch) - 1)
        ]
        return arch, unitaries, training_data, target_unitary

    def feedforward(
        self,
        arch: List[int],
        unitaries: List[Operator],
        samples: List[Tuple[Statevector, Statevector]],
    ) -> List[List[Statevector]]:
        """Propagate states through the layered unitary network."""
        stored: List[List[Statevector]] = []
        for state, _ in samples:
            layerwise: List[Statevector] = [state]
            current = state
            for unitary in unitaries:
                current = unitary @ current
                layerwise.append(current)
            stored.append(layerwise)
        return stored

    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        """Squared overlap of two pure states."""
        return abs(np.vdot(a.data, b.data)) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                fid = HybridSelfAttentionGraphQNN.state_fidelity(states[i], states[j])
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph
