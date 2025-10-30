"""Hybrid quantum neural network combining estimation and sampling.

The circuit reuses the same parameter set for both the EstimatorQNN and SamplerQNN,
allowing joint training and efficient reuse of quantum resources.  The estimator
uses a single Y‑observable on one qubit, while the sampler provides a full
probability distribution over two qubits."""
from __future__ import annotations

from qiskit.circuit import ParameterVector, Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp

class HybridEstimatorSamplerQNN:
    """Hybrid quantum neural network combining estimation and sampling."""
    def __init__(self) -> None:
        # Input and weight parameters
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 8)

        # Build a two‑qubit circuit
        self.circuit = QuantumCircuit(2)
        # Input encoding
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        # Entangling layer
        self.circuit.cx(0, 1)

        # Parameterized rotations
        for i, qubit in enumerate([0, 1]):
            self.circuit.ry(self.weight_params[i], qubit)
            self.circuit.rx(self.weight_params[i + 2], qubit)
        # Additional entanglement
        self.circuit.cx(0, 1)

        # Final rotations
        for i, qubit in enumerate([0, 1]):
            self.circuit.ry(self.weight_params[i + 4], qubit)
            self.circuit.rx(self.weight_params[i + 6], qubit)

        # Estimator observable (Y on first qubit)
        self.observable = SparsePauliOp.from_list([("Y" * self.circuit.num_qubits, 1)])

        # Estimator primitive
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = QEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.input_params[0]],
            weight_params=[self.weight_params[0]],
            estimator=self.estimator,
        )

        # Sampler primitive
        self.sampler = StatevectorSampler()
        self.sampler_qnn = QSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    def estimate(self, inputs: list[float]) -> float:
        """Return expectation value of the estimator observable."""
        return self.estimator_qnn.predict(inputs)

    def sample(self, inputs: list[float]) -> list[float]:
        """Return sampling probabilities from the sampler."""
        return self.sampler_qnn.predict(inputs)

__all__ = ["HybridEstimatorSamplerQNN"]
