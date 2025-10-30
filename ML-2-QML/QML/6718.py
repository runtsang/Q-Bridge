"""Hybrid quantum estimator with depth‑controlled ansatz and metadata.

This module provides a quantum variant of ``EstimatorQNN`` that shares
the same constructor signature and metadata attributes as the classical
implementation.  It builds a variational circuit with explicit input
encoding, depth‑controlled layers of Ry and CZ gates, and per‑qubit
Z‑observables.  The circuit can be wrapped by Qiskit’s
``EstimatorQNN`` class and trained with a classical optimizer.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

__all__ = ["EstimatorQNN", "build_classifier_circuit"]


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[
    QuantumCircuit,
    Iterable[ParameterVector],
    Iterable[ParameterVector],
    List[SparsePauliOp],
]:
    """
    Construct a layered variational ansatz with explicit encoding and
    variational parameters.

    Parameters
    ----------
    num_qubits: int
        Number of qubits / input features.
    depth: int
        Number of variational layers.

    Returns
    -------
    circuit: QuantumCircuit
        The variational circuit ready for training.
    input_params: Iterable[ParameterVector]
        Parameters that encode the classical inputs.
    weight_params: Iterable[ParameterVector]
        Variational parameters.
    observables: List[SparsePauliOp]
        Z observables on each qubit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Input encoding
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Entangling CZ gates
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


class EstimatorQNN:
    """
    Quantum estimator that mirrors the classical ``EstimatorQNN`` API.

    The constructor accepts the number of qubits, depth, and an optional
    primitive.  It builds the circuit, exposes the same metadata
    attributes (encoding, weight_sizes, observables) and wraps the
    Qiskit EstimatorQNN for training.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        depth: int = 1,
        estimator: StatevectorEstimator | None = None,
    ):
        self.num_qubits = num_qubits
        self.depth = depth

        (
            self.circuit,
            self.input_params,
            self.weight_params,
            self.observables,
        ) = build_classifier_circuit(num_qubits, depth)

        if estimator is None:
            estimator = StatevectorEstimator()

        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=estimator,
        )

        # Metadata analogous to the classical implementation
        self.weight_sizes = [len(p) for p in self.weight_params]
        self.encoding = list(range(num_qubits))

    def predict(self, inputs, weights):
        """
        Forward pass using the underlying Qiskit EstimatorQNN.

        Parameters
        ----------
        inputs: list or array-like
            Classical input values to be mapped to the encoding parameters.
        weights: list or array-like
            Variational weights for the circuit.

        Returns
        -------
        predictions: array-like
            Expected measurement values.
        """
        return self.estimator_qnn.predict(inputs, weights)

    def get_weight_sizes(self) -> List[int]:
        """Return the number of trainable parameters per variational layer."""
        return self.weight_sizes

    def get_encoding(self) -> List[int]:
        """Return the feature indices used for parameter encoding."""
        return self.encoding

    def get_observables(self) -> List[SparsePauliOp]:
        """Return the list of Z observables per qubit."""
        return self.observables
