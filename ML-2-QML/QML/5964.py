"""Quantum implementation of the EstimatorQNNHybrid using Qiskit."""

from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


class EstimatorQNNHybrid:
    """
    Wraps Qiskit’s EstimatorQNN to provide a quantum‑based regression model.
    Inputs and weights are encoded as rotation angles on a 3‑qubit circuit
    and a Pauli‑Y observable is measured to produce the output.
    """

    def __init__(self, input_dim: int = 2, weight_dim: int = 1) -> None:
        # Define circuit parameters
        self.input_params = [Parameter(f"input_{i}") for i in range(input_dim)]
        self.weight_params = [Parameter(f"weight_{i}") for i in range(weight_dim)]

        # Build the quantum circuit
        qc = QuantumCircuit(input_dim + weight_dim)
        for i, p in enumerate(self.input_params):
            qc.ry(p, i)
        for i, p in enumerate(self.weight_params):
            qc.rx(p, input_dim + i)

        # Observable for regression output
        observable = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)])

        # Instantiate the Qiskit EstimatorQNN
        estimator = StatevectorEstimator()
        self.model = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=estimator,
        )

    def predict(self, inputs: list[list[float]], weights: list[list[float]]) -> list[float]:
        """
        Parameters
        ----------
        inputs : list[list[float]]
            Input samples for the regression.
        weights : list[list[float]]
            Weight samples to be optimized during training.

        Returns
        -------
        list[float]
            Predicted values from the quantum circuit.
        """
        return self.model.predict(inputs, weights)

__all__ = ["EstimatorQNNHybrid"]
