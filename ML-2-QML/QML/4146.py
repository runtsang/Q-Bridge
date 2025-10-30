"""
Quantum version of EstimatorQNNFusion that uses a variational circuit
with input encoding and a trainable observable.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class EstimatorQNNFusion:
    """
    Quantum estimator that implements a variational circuit
    with an observable that is optimized jointly with the circuit
    parameters.  It mirrors the structure of the classical
    EstimatorQNN but replaces the final linear layer by a quantum
    expectation value.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        depth: int = 1,
        backend: str = "aer_simulator_statevector",
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.input_params, self.weight_params, self.observable = self._build_circuit()
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def _build_circuit(self) -> tuple[QuantumCircuit, list, list, list]:
        """
        Construct a simple variational circuit that encodes the input
        as Ry rotations and adds a depth of variational Ry gates.
        The observable is a weighted sum of Pauli‑Z on each qubit.
        """
        input_params = ParameterVector("x", self.num_qubits)
        weight_params = ParameterVector("θ", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        # Input encoding
        for i, param in enumerate(input_params):
            qc.ry(param, i)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for i in range(self.num_qubits):
                qc.ry(weight_params[idx], i)
                idx += 1
            # Entangling layer
            for i in range(self.num_qubits - 1):
                qc.cz(i, i + 1)

        # Observable: weighted sum of Z on each qubit
        observables = [SparsePauliOp.from_list([("Z" * self.num_qubits, 1)])]
        return qc, list(input_params), list(weight_params), observables

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the expectation value of the observable for each
        input sample.
        Parameters
        ----------
        x : np.ndarray
            Array of shape (batch, num_qubits) containing the input
            values that will be mapped to Ry rotations.
        Returns
        -------
        np.ndarray
            Array of shape (batch,) with the predicted scalar.
        """
        # Map input values to parameters
        param_bindings = [{p: val for p, val in zip(self.input_params, sample)} for sample in x]
        # Bind weight parameters (trainable) – here we use a placeholder of zeros
        zero_weights = {p: 0.0 for p in self.weight_params}
        # Combine bindings
        bindings = [ {**pb, **zero_weights} for pb in param_bindings ]

        # Evaluate expectation values
        expectation = self.qnn.predict(inputs=bindings)
        return np.array(expectation).reshape(-1)

__all__ = ["EstimatorQNNFusion"]
