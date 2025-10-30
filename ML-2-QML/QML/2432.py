"""Quantum classifier implementation using Qiskit.

The class exposes the same API as the classical counterpart:
`encoding`, `weight_sizes`, `observables` and a `forward` method
that returns expectation values of the observables.  Training is
performed by updating the `weight_params` of the variational ansatz
via a classical optimiser.

The circuit is constructed by `build_classifier_circuit`, which
mirrors the classical metadata and is compatible with the
EstimatorQNN interface from Qiskit Machine Learning.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as QiskitEstimator


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[
    QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]
]:
    """
    Construct a layered ansatz with explicit encoding and variational
    parameters.  The function returns the circuit, the encoding
    parameters, the weight parameters and a list of Pauli‑Z
    observables – identical to the classical metadata.

    Parameters
    ----------
    num_qubits:
        Number of qubits (equal to the dimensionality of the input).
    depth:
        Number of variational layers.

    Returns
    -------
    circuit:
        Qiskit QuantumCircuit instance.
    encoding:
        ParameterVector for data encoding.
    weight_params:
        ParameterVector for variational parameters.
    observables:
        List of SparsePauliOp objects to be measured.
    """
    encoding = ParameterVector("x", num_qubits)
    weight_params = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)

    # Data encoding – RX rotations
    for qubit, param in enumerate(encoding):
        qc.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weight_params[idx], qubit)
            idx += 1
        # Entangling layer
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    # Observables – Pauli‑Z on each qubit
    observables = [
        SparsePauliOp.from_list([("Z" * i + "I" * (num_qubits - i - 1), 1)])
        for i in range(num_qubits)
    ]

    return qc, encoding, weight_params, observables


class QuantumClassifierModel:
    """
    Quantum neural network classifier based on Qiskit EstimatorQNN.

    The class wraps an EstimatorQNN instance and exposes a `forward`
    method that returns expectation values of the observables.
    It also provides access to the trainable parameters via
    `weight_params`.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.circuit, self.encoding, self._weight_params, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        # Estimator for evaluating expectation values
        self.estimator = QiskitEstimator()
        # Wrap with EstimatorQNN for a convenient forward interface
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=list(self.encoding),
            weight_params=list(self._weight_params),
            estimator=self.estimator,
        )

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for a batch of input data.

        Parameters
        ----------
        input_data:
            2‑D array of shape (batch_size, num_qubits) containing
            real‑valued features.

        Returns
        -------
        np.ndarray:
            Expectation values of the observables for each input.
        """
        # Map inputs to encoding parameters
        param_bindings = [{param: val for param, val in zip(self.encoding, row)} for row in input_data]
        # Use EstimatorQNN to compute expectation values
        return self.qnn.evaluate(param_bindings)

    @property
    def weight_params(self) -> List[ParameterVector]:
        """Return the list of variational parameters."""
        return list(self._weight_params)

    @property
    def weight_sizes(self) -> List[int]:
        """Number of trainable parameters per weight vector."""
        return [len(p) for p in self._weight_params]


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
