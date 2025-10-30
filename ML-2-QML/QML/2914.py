from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, Operator, partial_trace, random_unitary
from qiskit.quantum_info import SparsePauliOp

class GraphQNNGen179:
    """
    Quantum graph neural network utilities with classical‑like interface.
    """

    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> Operator:
        """Generate a random unitary operator on `num_qubits` qubits."""
        mat = random_unitary(2 ** num_qubits).data
        return Operator(mat)

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> Statevector:
        """Generate a random pure state on `num_qubits` qubits."""
        dim = 2 ** num_qubits
        vec = np.random.normal(size=(dim,)) + 1j * np.random.normal(size=(dim,))
        vec /= np.linalg.norm(vec)
        return Statevector(vec)

    @staticmethod
    def random_training_data(unitary: Operator, samples: int) -> List[Tuple[Statevector, Statevector]]:
        """Produce input–output pairs by applying a target unitary to random states."""
        dataset: List[Tuple[Statevector, Statevector]] = []
        num_qubits = int(np.log2(unitary.data.shape[0]))
        for _ in range(samples):
            state = GraphQNNGen179._random_qubit_state(num_qubits)
            dataset.append((state, unitary @ state))
        return dataset

    @staticmethod
    def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[Operator]], List[Tuple[Statevector, Statevector]], Operator]:
        """Randomly construct a layered quantum circuit and data."""
        target_unitary = GraphQNNGen179._random_qubit_unitary(qnn_arch[-1])
        training_data = GraphQNNGen179.random_training_data(target_unitary, samples)

        unitaries: List[List[Operator]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: List[Operator] = []
            for output in range(num_outputs):
                op = GraphQNNGen179._random_qubit_unitary(num_inputs + 1)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return qnn_arch, unitaries, training_data, target_unitary

    @staticmethod
    def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[Operator]], layer: int, input_state: Statevector) -> Statevector:
        """Apply a single layer of the network, returning the reduced state."""
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        ancilla = Statevector.from_label('0' * num_outputs)
        state = input_state.tensor(ancilla)
        layer_unitary = unitaries[layer][0].data
        for gate in unitaries[layer][1:]:
            layer_unitary = gate.data @ layer_unitary
        full_unitary = Operator(layer_unitary)
        new_state = full_unitary @ state
        return partial_trace(new_state, list(range(num_inputs, num_inputs + num_outputs)))

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[Operator]], samples: Iterable[Tuple[Statevector, Statevector]]) -> List[List[Statevector]]:
        """Propagate quantum states through the network, storing all intermediate states."""
        stored: List[List[Statevector]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(qnn_arch)):
                current_state = GraphQNNGen179._layer_channel(qnn_arch, unitaries, layer, current_state)
                layerwise.append(current_state)
            stored.append(layerwise)
        return stored

    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        """Return the squared overlap between two pure quantum states."""
        return abs(a.data.conj().T @ b.data)[0, 0] ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[Statevector], threshold: float, *, secondary: Optional[float] = None, secondary_weight: float = 0.5) -> nx.Graph:
        """Construct a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen179.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Create a variational circuit with incremental data‑uploading and Pauli‑Z observables."""
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            circuit.rx(param, qubit)

        index = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[index], qubit)
                index += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
        return circuit, list(encoding), list(weights), observables
