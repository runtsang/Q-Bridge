"""GraphQNNGen330: quantum counterpart of the hybrid GNN.

The class offers the same public API as its classical sibling,
but internally operates on Qiskit objects: QuantumCircuit, Statevector,
SparsePauliOp.  The implementation reuses the fidelity‑based
graph construction and a simple variational ansatz for the
classifier.  The design keeps the quantum code lightweight
while exposing a familiar interface for experiments.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Union

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

State = Statevector
DataSample = Tuple[State, State]
NetworkArch = Sequence[int]
TrainingData = List[DataSample]


class GraphQNNGen330:
    """
    Quantum implementation of the hybrid graph neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer widths; the first entry is the input width.
    """

    def __init__(self, arch: NetworkArch) -> None:
        self.arch = list(arch)

    # ------------------------------------------------------------------
    # Helper utilities – adapted from the original QML seed
    # ------------------------------------------------------------------
    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> State:
        dim = 2 ** num_qubits
        matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(
            size=(dim, dim)
        )
        q, _ = np.linalg.qr(matrix)
        return State(q)

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> State:
        dim = 2 ** num_qubits
        amp = np.random.normal(size=dim) + 1j * np.random.normal(size=dim)
        amp /= np.linalg.norm(amp)
        return State(amp)

    # ------------------------------------------------------------------
    # Training data generation
    # ------------------------------------------------------------------
    @staticmethod
    def random_training_data(unitary: State, samples: int) -> TrainingData:
        data: TrainingData = []
        for _ in range(samples):
            state = GraphQNNGen330._random_qubit_state(unitary.num_qubits)
            data.append((state, unitary @ state))
        return data

    @classmethod
    def random_network(
        cls,
        arch: NetworkArch,
        samples: int,
    ) -> Tuple[NetworkArch, List[List[State]], TrainingData, State]:
        """
        Create a random variational circuit architecture and a
        matching training set.  The final layer implements a
        random unitary that serves as the target for supervised
        learning.
        """
        target = cls._random_qubit_unitary(arch[-1])
        training = cls.random_training_data(target, samples)

        unitaries: List[List[State]] = [[]]
        for layer in range(1, len(arch)):
            num_inputs = arch[layer - 1]
            num_outputs = arch[layer]
            ops: List[State] = []
            for output in range(num_outputs):
                op = cls._random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    # Embed the gate in a larger Hilbert space
                    op = State(np.kron(op.data, np.eye(2 ** (num_outputs - 1))))
                    # Swap the newly added qubit into the correct position
                    idx = num_inputs + output
                    label = list(op.label)
                    label[num_inputs], label[idx] = label[idx], label[num_inputs]
                    op = State.from_label("".join(label))
                ops.append(op)
            unitaries.append(ops)

        return list(arch), unitaries, training, target

    # ------------------------------------------------------------------
    # Fidelity utilities – identical to the classical implementation
    # ------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: State, b: State) -> float:
        """Absolute squared overlap between two pure states."""
        return abs((a.dag() @ b).data[0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[State],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(
            enumerate(states), 2
        ):
            fid = GraphQNNGen330.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    # Classifier construction – quantum version
    # ------------------------------------------------------------------
    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
    ) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """
        Return a layered ansatz with explicit feature encoding
        and variational parameters.  The signature matches the
        classical helper so that a hybrid training loop can
        interchange the two seamlessly.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        qc = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        return qc, list(encoding), list(weights), observables


__all__ = ["GraphQNNGen330"]
