"""Hybrid quantum estimator that embeds a self‑attention style entanglement into
a variational read‑out circuit.  The circuit consists of input rotations,
controlled‑Ry entanglement between neighbouring qubits (mimicking the
self‑attention pattern), and weight rotations before measuring an observable
on the last qubit.  The estimator is built on top of Qiskit Machine Learning's
EstimatorQNN, enabling gradient‑based optimisation of both the circuit
parameters and the output observable.

The class is compatible with the Aer simulator and can be used as a drop‑in
replacement for the classical hybrid model in a quantum‑classical workflow.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator
from qiskit import Aer, execute


class HybridEstimatorQNN:
    """
    Quantum hybrid estimator.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits in the circuit.
    """

    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")

        # Parameter vectors
        self.input_params = ParameterVector("x", length=n_qubits)
        self.attn_params = ParameterVector("a", length=n_qubits - 1)
        self.weight_params = ParameterVector("w", length=n_qubits)

        # Build the variational circuit
        self.circuit = QuantumCircuit(n_qubits)

        # Input encoding: Ry rotations
        for i in range(n_qubits):
            self.circuit.ry(self.input_params[i], i)

        # Self‑attention entanglement: controlled‑Ry between neighbours
        for i in range(n_qubits - 1):
            self.circuit.crx(self.attn_params[i], i, i + 1)

        # Weight rotations before measurement
        for i in range(n_qubits):
            self.circuit.rz(self.weight_params[i], i)

        # Observable: Pauli‑Y on the last qubit (scaled by 1 for simplicity)
        self.observable = SparsePauliOp.from_list([("Y" * n_qubits, 1)])

        # EstimatorQNN wrapper
        self.estimator = Estimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.input_params[i] for i in range(n_qubits)],
            weight_params=[self.weight_params[i] for i in range(n_qubits)],
            estimator=self.estimator,
        )

    def predict(
        self,
        inputs: np.ndarray,
        weights: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Evaluate the circuit for a single data point.

        Parameters
        ----------
        inputs : np.ndarray
            Input angles of shape (n_qubits,).
        weights : np.ndarray
            Weight angles of shape (n_qubits,).
        shots : int, optional
            Number of shots for expectation estimation.

        Returns
        -------
        np.ndarray
            Expectation value of the observable.
        """
        param_dict = {
            self.input_params[i]: inputs[i] for i in range(self.n_qubits)
        }
        param_dict.update(
            {self.weight_params[i]: weights[i] for i in range(self.n_qubits)}
        )

        # Use the EstimatorQNN predict method
        return self.estimator_qnn.predict(param_dict)

    def trainable_parameters(self) -> list[ParameterVector]:
        """
        Return all trainable parameter vectors for optimisation.

        Returns
        -------
        list[ParameterVector]
            List containing input, attention, and weight parameters.
        """
        return [self.input_params, self.attn_params, self.weight_params]


__all__ = ["HybridEstimatorQNN"]
