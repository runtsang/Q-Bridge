"""Quantum neural network estimator that supports entangled ansatz and multi‑qubit observables."""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as BaseEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
import torch
from typing import List, Tuple

class EstimatorQNN(BaseEstimatorQNN):
    """
    Extends Qiskit’s EstimatorQNN with a configurable entangling ansatz.
    Parameters:
        num_qubits : int
            Number of qubits in the circuit.
        depth : int
            Number of entangling layers.
        entangler_map : List[Tuple[int, int]]
            Qubit pairs to entangle in each layer.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        depth: int = 2,
        entangler_map: List[Tuple[int, int]] = None,
        input_params: List[Parameter] = None,
        weight_params: List[Parameter] = None,
        estimator: StatevectorEstimator = None,
        observables: SparsePauliOp = None,
    ) -> None:
        # Build the ansatz
        if entangler_map is None:
            entangler_map = [(i, i + 1) for i in range(num_qubits - 1)]
        circuit = QuantumCircuit(num_qubits)
        for layer in range(depth):
            # Input encoding
            for qubit in range(num_qubits):
                circuit.ry(Parameter(f"input_{qubit}_{layer}"), qubit)
            # Parameterized rotation
            for qubit in range(num_qubits):
                circuit.rx(Parameter(f"weight_{qubit}_{layer}"), qubit)
            # Entangling gates
            for q1, q2 in entangler_map:
                circuit.cx(q1, q2)
        # Default observable: sum of Z on each qubit
        if observables is None:
            paulis = [(f"Z" * num_qubits, 1.0)]
            observables = SparsePauliOp.from_list(paulis)
        # Default estimator
        if estimator is None:
            estimator = StatevectorEstimator()
        # Prepare parameter lists
        if input_params is None:
            input_params = [
                circuit.find_parameters(f"input_{q}_{0}") for q in range(num_qubits)
            ]
        if weight_params is None:
            weight_params = [
                circuit.find_parameters(f"weight_{q}_{0}") for q in range(num_qubits)
            ]
        super().__init__(
            circuit=circuit,
            observables=observables,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
        )

    def expectation(self, data: List[float]) -> List[float]:
        """
        Compute expectation values for a batch of input data.
        Each data point should be a list of length `num_qubits`.
        """
        return self.estimator.run(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            parameter_values=data,
        ).result().values

    def train(
        self,
        X: List[List[float]],
        y: List[float],
        epochs: int = 10,
        lr: float = 0.01,
    ) -> None:
        """
        Simple gradient‑descent training loop using the built‑in Estimator.
        """
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            opt.zero_grad()
            preds = torch.tensor(self.expectation(X), dtype=torch.float32)
            loss = torch.mean((preds - torch.tensor(y, dtype=torch.float32)) ** 2)
            loss.backward()
            opt.step()
            if epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch}: loss={loss.item():.4f}")

__all__ = ["EstimatorQNN"]
