"""Quantum graph‑based neural network utilities and classifier.

Combines the QNN helper and the quantum classifier builder from the
original seeds.  All API names match the classical counterpart for
cross‑compatibility, but the implementations use Qiskit state‑vector
simulation.  All data are represented by :class:`qiskit.quantum_info.Statevector` objects.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector, random_unitary
from qiskit.quantum_info import Statevector as QSState


class GraphQNN:
    """Quantum graph‑based neural network utilities and classifier.

    The API mirrors the classical GraphQNN class so that the same
    function names can be called from either side.  Internally the
    implementation uses Qiskit state‑vector simulation.  All data are
    represented by :class:`qiskit.quantum_info.Statevector` objects.
    """

    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> QSState:
        """Return a random unitary as a state‑vector (via matrix)."""
        dim = 2 ** num_qubits
        U = random_unitary(dim).data
        return QSState(U)

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> QSState:
        """Create a random pure state."""
        dim = 2 ** num_qubits
        vec = np.random.randn(dim) + 1j * np.random.randn(dim)
        vec /= np.linalg.norm(vec)
        return QSState(vec)

    @staticmethod
    def random_training_data(unitary: QSState, samples: int) -> List[Tuple[QSState, QSState]]:
        dataset: List[Tuple[QSState, QSState]] = []
        for _ in range(samples):
            state = GraphQNN._random_qubit_state(unitary.num_qubits)
            target = unitary.tensor(state)
            dataset.append((state, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: List[int], samples: int):
        """Generate a random QNN architecture with unitary layers."""
        target_unitary = GraphQNN._random_qubit_unitary(qnn_arch[-1])
        training_data = GraphQNN.random_training_data(target_unitary, samples)

        unitaries: List[List[QSState]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: List[QSState] = []
            for _ in range(num_outputs):
                op = GraphQNN._random_qubit_unitary(num_inputs + 1)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return qnn_arch, unitaries, training_data, target_unitary

    @staticmethod
    def _partial_trace(state: QSState, keep: Sequence[int]) -> QSState:
        """Return the partial trace over all qubits except ``keep``."""
        all_qubits = list(range(state.num_qubits))
        discard = [q for q in all_qubits if q not in keep]
        return state.partial_trace(discard)

    @staticmethod
    def _layer_channel(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[QSState]],
        layer: int,
        input_state: QSState,
    ) -> QSState:
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        # Append a zero state for the extra qubit(s)
        zero = QSState.from_label("0" * num_outputs)
        state = input_state.tensor(zero)
        # Apply the first unitary
        unitary = unitaries[layer][0]
        new_state = unitary.tensor(state)
        # Apply remaining unitaries sequentially
        for gate in unitaries[layer][1:]:
            new_state = gate.tensor(new_state)
        # Partial trace over input qubits
        keep = list(range(num_inputs, num_inputs + num_outputs))
        return GraphQNN._partial_trace(new_state, keep)

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[QSState]],
        samples: Iterable[Tuple[QSState, QSState]],
    ) -> List[List[QSState]]:
        stored: List[List[QSState]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current = sample
            for layer in range(1, len(qnn_arch)):
                current = GraphQNN._layer_channel(qnn_arch, unitaries, layer, current)
                layerwise.append(current)
            stored.append(layerwise)
        return stored

    @staticmethod
    def state_fidelity(a: QSState, b: QSState) -> float:
        """Squared overlap between two pure states."""
        return abs(np.vdot(a.data, b.data)) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[QSState],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int, depth: int
    ) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            circuit.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        return circuit, list(encoding), list(weights), observables


__all__ = [
    "GraphQNN",
]
