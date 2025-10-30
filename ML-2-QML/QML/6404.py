"""Hybrid quantum regressor implemented with Qiskit.

The class mirrors the PyTorch implementation:
  * Classical input features are encoded into RY rotations.
  * Trainable RX rotations act as the variational part.
  * Measurement of a multi‑qubit Pauli‑Z observable yields the prediction.
"""

import numpy as np
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class HybridRegressor:
    """Hybrid quantum regression network using Qiskit EstimatorQNN."""
    def __init__(self, num_features: int, num_qubits: int = 4):
        self.num_features = num_features
        self.num_qubits = num_qubits

        # Input parameters – one per feature
        self.input_params = [Parameter(f"x{i}") for i in range(num_features)]
        # Trainable weight parameters – one per qubit
        self.weight_params = [Parameter(f"w{i}") for i in range(num_qubits)]

        # Build the parameterised circuit
        self.circuit = QuantumCircuit(num_qubits)
        # Simple feature encoding: RY with input parameters
        for i, p in enumerate(self.input_params):
            self.circuit.ry(p, i % num_qubits)
        # Variational part: trainable RX rotations
        for i, w in enumerate(self.weight_params):
            self.circuit.rx(w, i)

        # Observable: product of Z on all qubits
        self.observable = SparsePauliOp.from_list([("Z" * num_qubits, 1)])

        # Estimator instance
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=estimator,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return expectation values for the batch X."""
        # Build input dictionaries for the estimator
        input_dicts = [{f"x{i}": val for i, val in enumerate(sample)} for sample in X]
        return self.estimator_qnn.predict(input_dicts)

__all__ = ["HybridRegressor"]
