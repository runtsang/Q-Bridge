"""Hybrid fraud detection model – quantum implementation.

The QML side implements a depth‑encoded variational circuit
similar to the Qiskit example in QuantumClassifierModel.py.
It accepts a 2‑D feature vector, encodes it with RX gates,
applies a trainable ansatz with Ry and CZ layers,
and measures the expectation of Z on each qubit.
The mean of these expectations is returned as the fraud score.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Tuple, Iterable


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], list[SparsePauliOp]]:
    """Construct a layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits:
        Number of qubits (features).
    depth:
        Depth of the variational layers.

    Returns
    -------
    circuit:
        Parameterised Qiskit circuit.
    encoding:
        ParameterVector for data encoding (RX).
    weights:
        ParameterVector for variational Ry gates.
    observables:
        List of Z‑Pauli operators for measurement.
    """
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
    return circuit, encoding, weights, observables


class FraudDetectionHybrid:
    """A Qiskit‑based quantum fraud‑detection model.

    The class mirrors the classical API but operates entirely on a
    parameterised quantum circuit.  It exposes a ``run`` method that
    accepts a NumPy array of shape (2,) and returns the fraud probability.
    """

    def __init__(self, depth: int = 2, backend=None):
        """
        Parameters
        ----------
        depth:
            Depth of the variational layers.
        backend:
            Qiskit backend to execute the circuit; if None, the Aer
            state‑vector simulator is used.
        """
        self.num_qubits = 2
        self.depth = depth
        self.backend = backend or Aer.get_backend("aer_simulator_statevector")

        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            self.num_qubits, self.depth
        )
        # Random initial parameters
        self.params = {
            "x": np.random.uniform(0, 2 * np.pi, self.num_qubits),
            "theta": np.random.uniform(0, 2 * np.pi, self.num_qubits * self.depth),
        }

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a single sample.

        Parameters
        ----------
        data:
            1‑D array of length 2 containing feature values in [0, 1].

        Returns
        -------
        float:
            Fraud probability between 0 and 1.
        """
        # Bind data and variational parameters
        bind_dict = {self.encoding[i]: data[i] for i in range(self.num_qubits)}
        bind_dict.update({self.weights[i]: self.params["theta"][i] for i in range(len(self.weights))})

        bound_circ = self.circuit.bind_parameters(bind_dict)

        job = execute(bound_circ, self.backend, shots=1024)
        result = job.result()
        # Evaluate expectation of each Z observable
        expz = result.get_expectation_value(self.observables[0])
        # For two qubits, take mean of |Z| expectation values
        expz += result.get_expectation_value(self.observables[1])
        expz /= self.num_qubits
        # Map expectation from [-1, 1] to [0, 1]
        return 0.5 * (expz + 1)

    def set_params(self, params: dict) -> None:
        """Update the variational parameters."""
        self.params.update(params)

    def get_params(self) -> dict:
        """Return current parameters."""
        return self.params


__all__ = ["FraudDetectionHybrid"]
