"""Quantum regression model with a two‑qubit entangling variational circuit.

The circuit uses parameter‑shift estimators and a Pauli‑Z observable on both qubits.
It can be seamlessly substituted for the classical EstimatorQNN in hybrid pipelines.
"""

from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator


class EstimatorQNN:
    """Two‑qubit variational circuit with entanglement and parameter‑shift estimator."""

    def __init__(self) -> None:
        # Define parameters for inputs and weights
        self.input_params = [Parameter("input1"), Parameter("input2")]
        self.weight_params = [Parameter("weight1"), Parameter("weight2")]

        # Build variational circuit
        qc = QuantumCircuit(2)
        # Entangling layer
        qc.h(0)
        qc.cx(0, 1)
        # Parameterized rotations
        qc.ry(self.input_params[0], 0)
        qc.ry(self.input_params[1], 1)
        qc.rx(self.weight_params[0], 0)
        qc.rx(self.weight_params[1], 1)
        # Additional entanglement
        qc.cx(1, 0)

        self.circuit = qc

        # Observable: Z⊗Z
        observable = Pauli("ZZ")
        self.observables = SparsePauliOp.from_list([(observable.to_label(), 1)])

        # Statevector estimator for parameter‑shift evaluation
        self.estimator = StatevectorEstimator()

        # Wrap into Qiskit EstimatorQNN
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def __call__(self, inputs: dict[str, float]) -> float:
        """Evaluate the quantum model for a single input sample.

        Args:
            inputs: Mapping from input parameter names to numerical values.

        Returns:
            The expectation value of the observable.
        """
        return float(self.estimator_qnn(inputs))

    def parameters(self) -> list[Parameter]:
        """Return all trainable parameters."""
        return self.weight_params


__all__ = ["EstimatorQNN"]
