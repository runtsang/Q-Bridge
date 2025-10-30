"""Quantum neural network estimator based on Qiskit’s EstimatorQNN.

The circuit is a simple variational ansatz that encodes a single input feature and a weight parameter.
An observable of Pauli‑Y is measured to produce a scalar expectation value used as the regression output.
The implementation is intentionally lightweight to allow integration with classical optimizers while
retaining the expressive power of a quantum circuit.

The design mirrors the EstimatorQNN example but is extended with a small quantum kernel that
acts on two qubits, enabling richer feature mapping without changing the overall interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

__all__ = ["HybridEstimatorQNN"]


class QuantumEstimatorCircuit(QuantumCircuit):
    """Variational circuit with a small quantum kernel for feature mapping."""

    def __init__(self, input_param: Parameter, weight_param: Parameter) -> None:
        super().__init__(2)
        # Encode 2‑qubit kernel: apply Ry on both qubits, then a controlled‑Rx
        self.h(0)
        self.h(1)
        self.ry(input_param, 0)
        self.ry(input_param, 1)
        self.cx(0, 1)
        self.rx(weight_param, 0)
        self.rx(weight_param, 1)
        self.cx(1, 0)


def make_estimator_qnn() -> EstimatorQNN:
    """Instantiate the Qiskit EstimatorQNN with a custom circuit."""
    input_param = Parameter("input")
    weight_param = Parameter("weight")

    qc = QuantumEstimatorCircuit(input_param, weight_param)

    # Observable: Pauli‑Y on the second qubit
    observable = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)])

    estimator = StatevectorEstimator()

    return EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[input_param],
        weight_params=[weight_param],
        estimator=estimator,
    )


class HybridEstimatorQNN(nn.Module):
    """Wrapper around Qiskit’s EstimatorQNN to expose a torch‑friendly interface."""

    def __init__(self) -> None:
        super().__init__()
        self.estimator_qnn = make_estimator_qnn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1) representing a single feature per example.

        Returns
        -------
        torch.Tensor
            Regression output of shape (B, 1).
        """
        with torch.no_grad():
            # Convert to numpy and run the Qiskit estimator
            inputs = x.detach().cpu().numpy().reshape(-1, 1)
            outputs = self.estimator_qnn.predict(inputs)
            return torch.from_numpy(outputs).float().to(x.device).unsqueeze(-1)
