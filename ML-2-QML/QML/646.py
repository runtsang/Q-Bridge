"""Quantum-enhanced regressor based on a two‑qubit entangled circuit."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
import numpy as np


class EstimatorQNNEnhanced(EstimatorQNN):
    """
    A variational quantum neural network that mirrors the classical
    EstimatorQNNEnhanced API.

    Features
    --------
    - Two‑qubit circuit with parameterised Ry, Rz, and CNOT layers
    - Separate input and weight parameters
    - Expectation values of {Z⊗I, I⊗Z, Z⊗Z} as observables
    - Uses the StatevectorEstimator backend for exact simulation

    Parameters
    ----------
    input_dim : int, default 2
        Number of real‑valued input features.
    weight_dim : int, default 2
        Number of trainable weight parameters.
    """

    def __init__(self, input_dim: int = 2, weight_dim: int = 2) -> None:
        # Build the variational circuit
        self.input_params = ParameterVector("x", input_dim)
        self.weight_params = ParameterVector("w", weight_dim)

        qc = QuantumCircuit(2)
        # Prepare entanglement
        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)

        # Parameterised rotations per qubit
        for i in range(2):
            qc.ry(self.input_params[i], i)
            qc.rz(self.input_params[i], i)
            qc.ry(self.weight_params[i], i)
            qc.rz(self.weight_params[i], i)

        # Additional entangling layer
        qc.cx(1, 0)

        # Define observables
        observables = [
            SparsePauliOp.from_list([("Z" * qc.num_qubits, 1)]),  # Z⊗Z
            SparsePauliOp.from_list([("Z", 1)]),                 # Z on qubit 0
            SparsePauliOp.from_list([("I", 1)]),                 # Identity (bias)
        ]

        # Initialise the parent EstimatorQNN with the constructed circuit
        super().__init__(
            circuit=qc,
            observables=observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=StatevectorEstimator(),
        )

    @staticmethod
    def example_inputs() -> np.ndarray:
        """
        Generate a small batch of example inputs for quick testing.

        Returns
        -------
        np.ndarray
            Shape (4, 2) with values in [-π, π].
        """
        return np.random.uniform(-np.pi, np.pi, size=(4, 2))


__all__ = ["EstimatorQNNEnhanced"]
