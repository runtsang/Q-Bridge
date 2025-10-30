"""Quantum kernel estimator that mirrors the classical hybrid architecture.

The quantum circuit encodes two input features into rotation angles on a
four‑qubit device, followed by a trainable entangling layer.  The observable
is a product of Pauli‑Y operators, and the estimator uses Qiskit’s
StatevectorEstimator to evaluate the overlap between input and support
vectors.  This construction parallels the classical RBF kernel in the ML
module, enabling a direct performance comparison."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.providers.fake_provider import FakeVigo
from qiskit.primitives import Estimator as QuantumEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals


class QuantumKernelAnsatz(QuantumCircuit):
    """
    Parameterised circuit that encodes two classical inputs and a single
    trainable weight on four qubits.  The encoding uses Ry rotations and
    the trainable layer consists of a sequence of CNOTs and Rz gates.
    """
    def __init__(self) -> None:
        self.qreg = QuantumRegister(4, "q")
        super().__init__(self.qreg)
        # Input parameters
        self.input1 = Parameter("x1")
        self.input2 = Parameter("x2")
        # Weight parameters
        self.w1 = Parameter("w1")
        self.w2 = Parameter("w2")
        self.w3 = Parameter("w3")
        self.w4 = Parameter("w4")

        # Data encoding
        self.ry(self.input1, 0)
        self.ry(self.input2, 1)
        self.ry(self.input1, 2)
        self.ry(self.input2, 3)

        # Trainable entangling layer
        self.cx(0, 1)
        self.rx(self.w1, 1)
        self.cx(1, 2)
        self.rx(self.w2, 2)
        self.cx(2, 3)
        self.rx(self.w3, 3)
        self.cx(3, 0)
        self.rx(self.w4, 0)


class EstimatorQNN(QiskitEstimatorQNN):
    """
    Wrapper around Qiskit’s EstimatorQNN that configures a custom quantum
    kernel circuit.  The class preserves the original API while adding
    support for a 4‑qubit kernel with trainable weights.
    """
    def __init__(self,
                 input_dim: int = 2,
                 kernel_dim: int = 8,
                 backend: str = "statevector_simulator") -> None:
        super().__init__()
        self.circuit = QuantumKernelAnsatz()
        # Observable: product of Y on all qubits
        self.observable = SparsePauliOp.from_list([("Y" * self.circuit.num_qubits, 1)])
        self.input_params = [self.circuit.input1,
                             self.circuit.input2]
        self.weight_params = [self.circuit.w1,
                              self.circuit.w2,
                              self.circuit.w3,
                              self.circuit.w4]
        # Define support vectors as trainable weights
        self.support = torch.nn.Parameter(torch.randn(kernel_dim, 4))
        # Use a simple backend for demonstration
        self.estimator = QuantumEstimator(backend=backend)

        # Build the EstimatorQNN instance
        self.estimator_qnn = super().__init__(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel between the input features and the support vectors
        using the quantum circuit.  The support vectors are treated as
        additional weight parameters in the circuit.
        """
        # Expand inputs to match support dimension
        batch_size = inputs.shape[0]
        # Create a list of circuits with different support vector values
        results = []
        for sv in self.support:
            # Bind parameters: x1, x2 from inputs; w1-w4 from support vector
            bound = self.circuit.bind_parameters({
                self.circuit.input1: inputs[:, 0],
                self.circuit.input2: inputs[:, 1],
                self.circuit.w1: sv[0],
                self.circuit.w2: sv[1],
                self.circuit.w3: sv[2],
                self.circuit.w4: sv[3]
            })
            # Evaluate expectation value
            exp = self.estimator.run(
                circuits=[bound],
                observables=[self.observable]
            ).result().values[0]
            results.append(exp)
        kernel_matrix = torch.stack(results, dim=-1)
        # Linear read‑out
        return torch.mean(kernel_matrix, dim=-1).unsqueeze(-1)


def EstimatorQNN() -> EstimatorQNN:
    """Return a quantum‑kernel EstimatorQNN with default configuration."""
    return EstimatorQNN()


__all__ = ["EstimatorQNN"]
