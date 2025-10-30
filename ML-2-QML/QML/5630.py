"""GraphQNNHybrid: quantum implementation of graph-based quantum neural networks.

This module implements the same API as the classical version but
uses Qiskit for circuit construction and simulation.  It supports:

* Random generation of a layered unitary network.
* Random training data obtained by applying the target unitary to
  randomly sampled basis states.
* Forward propagation through the network via statevector simulation.
* Fidelity‑based graph construction of intermediate states.
* A quantum classifier ansatz with explicit encoding and variational
  parameters.

The design mirrors the classical API so that experiments can be
run on either back‑end without code changes.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, Operator
from qiskit.opflow import PauliSumOp


class GraphQNNHybrid:
    """Class encapsulating the quantum graph‑based neural network API."""

    @staticmethod
    def _random_unitary(num_qubits: int) -> np.ndarray:
        """Generate a Haar‑distributed random unitary matrix."""
        dim = 2 ** num_qubits
        z = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
        q, _ = np.linalg.qr(z)
        return q

    @staticmethod
    def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[QuantumCircuit]], List[Tuple[Statevector, Statevector]], Operator]:
        """Randomly generate a layered unitary network and training data."""
        target_unitary = Operator(GraphQNNHybrid._random_unitary(qnn_arch[-1]))
        training_data = GraphQNNHybrid.random_training_data(target_unitary, samples)

        circuits: List[List[QuantumCircuit]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_circuits: List[QuantumCircuit] = []

            for output in range(num_outputs):
                qc = QuantumCircuit(num_inputs + 1)
                qc.unitary(GraphQNNHybrid._random_unitary(num_inputs + 1), qc.qubits, inplace=True)
                if num_outputs > 1:
                    qc.swap(num_inputs, num_inputs + output)
                layer_circuits.append(qc)

            circuits.append(layer_circuits)

        return qnn_arch, circuits, training_data, target_unitary

    @staticmethod
    def random_training_data(unitary: Operator, samples: int) -> List[Tuple[Statevector, Statevector]]:
        """Generate training data by applying a fixed unitary to random basis states."""
        dataset: List[Tuple[Statevector, Statevector]] = []
        num_qubits = int(np.log2(unitary.dim))
        for _ in range(samples):
            basis_index = np.random.randint(0, 2 ** num_qubits)
            state_in = Statevector.from_label(format(basis_index, f"0{num_qubits}b"))
            state_out = unitary.evolve(state_in)
            dataset.append((state_in, state_out))
        return dataset

    @staticmethod
    def _layer_apply(circuits: List[QuantumCircuit], state: Statevector) -> Statevector:
        """Apply a single layer of unitaries to the current state."""
        op = Operator(1.0)
        for qc in circuits:
            op = op @ Operator(qc)
        return op.evolve(state)

    @staticmethod
    def feedforward(qnn_arch: List[int], circuits: List[List[QuantumCircuit]],
                    samples: Iterable[Tuple[Statevector, Statevector]]) -> List[List[Statevector]]:
        """Propagate each sample through the quantum network."""
        outputs: List[List[Statevector]] = []
        for state_in, _ in samples:
            layerwise: List[Statevector] = [state_in]
            current = state_in
            for layer in range(1, len(qnn_arch)):
                current = GraphQNNHybrid._layer_apply(circuits[layer], current)
                layerwise.append(current)
            outputs.append(layerwise)
        return outputs

    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        """Compute the squared overlap between two pure statevectors."""
        return float(abs(a.data.conj().dot(b.data)) ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Statevector], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Return a weighted graph where edges represent high‑fidelity state pairs."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[PauliSumOp]]:
        """Construct a layered variational ansatz with explicit encoding."""
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

        observables = [PauliSumOp.from_list([("I" * i + "Z" + "I" * (num_qubits - i - 1), 1)]) for i in range(num_qubits)]
        return qc, list(encoding), list(weights), observables


__all__ = [
    "GraphQNNHybrid",
]
