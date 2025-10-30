"""Hybrid quantum‑graph classifier – quantum implementation."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
import networkx as nx

# Local utilities – identical signatures to the original seeds
from GraphQNN import (
    feedforward as _feedforward,
    fidelity_adjacency as _fidelity_adjacency,
    random_network as _random_network,
    random_training_data as _random_training_data,
    state_fidelity as _state_fidelity,
)
from FastBaseEstimator import FastBaseEstimator


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        Variational circuit.
    encoding : List[ParameterVector]
        Encoding parameters.
    weights : List[ParameterVector]
        Variational parameters.
    observables : List[SparsePauliOp]
        Measurement operators.
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


class HybridQuantumGraphClassifier:
    """
    Quantum side of the hybrid classifier.
    Builds a variational circuit, propagates quantum states,
    and constructs a fidelity‑based graph of intermediate states.
    """

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits
        (
            self.circuit,
            self.encoding,
            self.weights,
            self.observables,
        ) = self.build_classifier_circuit(depth=2)
        self.estimator = FastBaseEstimator(self.circuit)

    def random_training_data(self, samples: int = 200) -> List[Tuple[Statevector, Statevector]]:
        """Generate synthetic training data using a random target unitary."""
        target_unitary = _random_network([self.num_qubits], samples)[-1]
        return _random_training_data(target_unitary, samples)

    def train(self, data: List[Tuple[Statevector, Statevector]], shots: int = 1024) -> None:
        """Placeholder for a variational training loop."""
        # In practice, one would use a gradient‑based optimiser.
        pass

    def graph_of_hidden_states(self, data: Iterable[Tuple[Statevector, Statevector]]) -> nx.Graph:
        """Return a graph where nodes are hidden quantum states and edges are weighted by fidelity."""
        states = _feedforward([self.num_qubits], [self.circuit], data)
        flat_states = [s for layer in states for s in layer[1:]]
        return _fidelity_adjacency(flat_states, threshold=0.9)

    def evaluate(self, inputs: Iterable[Statevector]) -> List[complex]:
        """Return expectation values for each input state."""
        return self.estimator.evaluate(self.observables, [inp.data for inp in inputs])

    def build_classifier_circuit(self, depth: int = 2) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """Return the variational circuit, encoding, weight parameters, and observables."""
        return self.circuit, self.encoding, self.weights, self.observables


__all__ = ["HybridQuantumGraphClassifier", "build_classifier_circuit"]
