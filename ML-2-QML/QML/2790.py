"""Hybrid quantum estimator.

The quantum circuit encodes input features using Ry rotations and then
applies a small trainable variational layer.  The circuit is wrapped
inside a Qiskit EstimatorQNN which returns a single expectation value.
The design mirrors the classical hybrid by using a quantum kernel
implementation as the observable: the overlap of two encoded states
is measured via a Y observable on the first qubit.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
import torch

class HybridEstimator:
    """
    Quantum neural network estimator.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the classical input.
    weight_dim : int
        Number of trainable parameters in the variational layer.
    """

    def __init__(self, input_dim: int, weight_dim: int = 4) -> None:
        self.input_dim = input_dim
        self.weight_dim = weight_dim

        # Parameters for data reuploading
        self.input_params = [
            Parameter(f"inp_{i}") for i in range(input_dim)
        ]
        # Trainable weights for the variational layer
        self.weight_params = [
            Parameter(f"w_{i}") for i in range(weight_dim)
        ]

        # Build the circuit
        self.circuit = QuantumCircuit(input_dim)
        # Data encoding with Ry gates
        for i, p in enumerate(self.input_params):
            self.circuit.ry(p, i)
        # Variational layer: simple two‑qubit entanglement followed by Ry
        for i in range(weight_dim):
            self.circuit.ry(self.weight_params[i % weight_dim], i % input_dim)
            if i < input_dim - 1:
                self.circuit.cx(i, (i + 1) % input_dim)

        # Observable: Y on the first qubit
        self.observable = SparsePauliOp.from_list([("Y" + "I" * (input_dim - 1), 1.0)])

        # Estimator primitive
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def __call__(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the quantum circuit for a batch of inputs.

        Parameters
        ----------
        input_data : torch.Tensor
            Tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Output of shape (batch, 1).
        """
        # Convert to numpy for the Estimator
        inputs_np = input_data.numpy()
        params = {p: inputs_np[:, i] for i, p in enumerate(self.input_params)}
        # The EstimatorQNN expects a dictionary of parameters for each input
        # It automatically handles batching.
        results = self.estimator_qnn.evaluate(params)
        # Convert back to torch
        return torch.tensor(results, dtype=torch.float32).unsqueeze(-1)

__all__ = ["HybridEstimator"]
