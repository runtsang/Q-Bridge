"""Quantum counterpart of GraphQNNHybrid.

Provides random network generation, feed‑forward propagation,
fidelity‑based adjacency, and an estimator API with optional shot noise.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# --------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Return a random Haar‑distributed unitary matrix."""
    dim = 2 ** num_qubits
    from scipy import linalg, random as sl_random

    mat = sl_random.normal(size=(dim, dim)) + 1j * sl_random.normal(size=(dim, dim))
    q, _ = linalg.qr(mat)
    return q

def _random_qubit_state(num_qubits: int) -> Statevector:
    """Return a random pure state vector."""
    dim = 2 ** num_qubits
    vec = np.random.normal(size=(dim,)) + 1j * np.random.normal(size=(dim,))
    vec /= np.linalg.norm(vec)
    return Statevector(vec)

def _random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[Statevector, Statevector]]:
    dataset: List[Tuple[Statevector, Statevector]] = []
    for _ in range(samples):
        inp = _random_qubit_state(unitary.shape[0].bit_length() - 1)
        out = Statevector(unitary @ inp.data)
        dataset.append((inp, out))
    return dataset

# --------------------------------------------------------------------- #
# Core class
# --------------------------------------------------------------------- #
class GraphQNNHybrid:
    """Quantum graph‑neural‑network with estimator support."""

    def __init__(self, qnn_arch: Sequence[int], device: str | None = None) -> None:
        self.arch = list(qnn_arch)
        self.device = device
        self.unitaries: List[List[np.ndarray]] | None = None
        self.target_unitary: np.ndarray | None = None

    def initialize_random_network(self, samples: int) -> List[Tuple[Statevector, Statevector]]:
        """Generate a random quantum network and training data."""
        self.target_unitary = _random_qubit_unitary(self.arch[-1])
        training_data = _random_training_data(self.target_unitary, samples)

        unitaries: List[List[np.ndarray]] = [[]]
        for layer in range(1, len(self.arch)):
            in_f = self.arch[layer - 1]
            out_f = self.arch[layer]
            layer_ops: List[np.ndarray] = []
            for output in range(out_f):
                op = _random_qubit_unitary(in_f + 1)
                if out_f > 1:
                    id_other = np.eye(2 ** (out_f - 1), dtype=complex)
                    op = np.kron(op, id_other)
                    perm = list(range(op.shape[0]))
                    perm[in_f], perm[in_f + output] = perm[in_f + output], perm[in_f]
                    op = op[perm, :][:, perm]
                layer_ops.append(op)
            unitaries.append(layer_ops)

        self.unitaries = unitaries
        return training_data

    # ------------------------------------------------------------------ #
    # Feed‑forward propagation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _layer_channel(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[np.ndarray]],
        layer: int,
        input_state: Statevector,
    ) -> Statevector:
        in_f = qnn_arch[layer - 1]
        out_f = qnn_arch[layer]
        ancilla = Statevector(np.zeros(2 ** out_f, dtype=complex))
        ancilla.data[0] = 1.0
        state = Statevector(np.kron(input_state.data, ancilla.data))

        layer_unitary = unitaries[layer][0]
        for gate in unitaries[layer][1:]:
            layer_unitary = gate @ layer_unitary

        new_state = Statevector(layer_unitary @ state.data)
        # keep only the first in_f qubits
        return new_state.copy()

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[np.ndarray]],
        samples: Iterable[Tuple[Statevector, Statevector]],
    ) -> List[List[Statevector]]:
        stored: List[List[Statevector]] = []
        for inp, _ in samples:
            layerwise = [inp]
            current = inp
            for layer in range(1, len(qnn_arch)):
                current = GraphQNNHybrid._layer_channel(qnn_arch, unitaries, layer, current)
                layerwise.append(current)
            stored.append(layerwise)
        return stored

    # ------------------------------------------------------------------ #
    # Fidelity‑based graph construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        return abs((a.data.conj() @ b.data)) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------ #
    # Estimator API
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        if self.unitaries is None or self.target_unitary is None:
            raise RuntimeError("Network not initialised – call ``initialize_random_network`` first.")

        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            # Construct a parameterised circuit that mimics the network
            num_params = sum(self.arch)
            if len(params)!= num_params:
                raise ValueError("Parameter count mismatch for the network.")
            circ = QuantumCircuit(num_params)
            # bind parameters to a simple RY gate on each qubit
            for qubit, val in enumerate(params):
                circ.ry(val, qubit)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                complex(
                    val.real + rng.normal(0, max(1e-6, 1 / shots)),
                    val.imag + rng.normal(0, max(1e-6, 1 / shots)),
                )
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

    # ------------------------------------------------------------------ #
    # Convenience utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sample_random_parameters(
        qnn_arch: Sequence[int], num_samples: int, seed: int | None = None
    ) -> List[List[float]]:
        rng = random.Random(seed)
        return [
            [rng.uniform(-np.pi, np.pi) for _ in range(f)]
            for f in qnn_arch
        ][:num_samples]

    def __repr__(self) -> str:
        return f"<GraphQNNHybrid arch={self.arch} device={self.device}>"
