"""
Quantum‑classical hybrid estimator that replaces the toy circuit with a 2‑qubit variational ansatz.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class EstimatorQNN:
    """
    Wrapper that builds a 2‑qubit variational circuit, maps inputs/weights to parameters,
    and exposes a predict method compatible with the classical API.
    """
    def __init__(self) -> None:
        # Parameter vectors
        self.input_params = ParameterVector("x", 2)
        self.weight_params = ParameterVector("w", 4)

        # Build circuit
        self.circuit = QuantumCircuit(2)
        self.circuit.h([0, 1])  # initial superposition

        # Encode inputs
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)

        # Entanglement
        self.circuit.cz(0, 1)

        # Variational layers
        for i, w in enumerate(self.weight_params):
            qubit = i % 2
            self.circuit.ry(w, qubit)

        # Observable (expectation of Z⊗Z)
        self.observable = SparsePauliOp.from_list([("ZZ", 1)])

        # Estimator backend
        self.estimator = StatevectorEstimator()

        # Instantiate Qiskit EstimatorQNN
        self.qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the quantum circuit on a batch of 2‑dimensional input vectors.
        """
        return self.qnn.predict(inputs)

__all__ = ["EstimatorQNN"]
