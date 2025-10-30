"""Quantum estimator that mirrors the classical EstimatorQNN.

The quantum circuit encodes two input features using RY rotations and
applies a two‑qubit entangling layer.  Trainable rotation angles are
treated as weight parameters.  The circuit is evaluated with a
Pauli‑Z⊗Z observable using a state‑vector estimator.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

class EstimatorQNN(QiskitEstimatorQNN):
    """Quantum neural network estimator for regression."""
    
    def __init__(self,
                 input_dim: int = 2,
                 weight_dim: int = 4,
                 dropout: float = 0.0) -> None:
        # Define parameters
        input_params = [Parameter(f"x{i}") for i in range(input_dim)]
        weight_params = [Parameter(f"w{i}") for i in range(weight_dim)]

        # Build circuit
        qc = QuantumCircuit(input_dim)
        # Data encoding
        for i, p in enumerate(input_params):
            qc.ry(p, i)
        # Entangling layer
        qc.cx(0, 1)
        # Parameterised rotations
        for i, p in enumerate(weight_params):
            qc.rz(p, i % input_dim)
        # Observable
        observable = SparsePauliOp.from_list([("ZZ", 1.0)])

        # Create estimator
        estimator = StatevectorEstimator()
        super().__init__(circuit=qc,
                         observables=observable,
                         input_params=input_params,
                         weight_params=weight_params,
                         estimator=estimator)

    @staticmethod
    def build_default() -> "EstimatorQNN":
        """Convenience constructor that matches the original seed API."""
        return EstimatorQNN()

__all__ = ["EstimatorQNN"]
