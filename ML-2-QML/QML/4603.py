"""Hybrid quantum expectation head for the classical regression network.

Features:
- Two‑qubit circuit with Ry rotations (input encoding) and a trainable Rx gate (weight).
- Uses Qiskit Machine Learning EstimatorQNN for efficient back‑propagation.
- Wraps the Qiskit estimator in a torch.nn.Module so it can be composed with the classical head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN


class QuantumEstimatorQNN(nn.Module):
    """Parameter‑ised two‑qubit quantum circuit serving as an expectation head."""

    def __init__(self, n_qubits: int = 2, backend=None) -> None:
        super().__init__()
        if n_qubits!= 2:
            raise ValueError("QuantumEstimatorQNN currently supports exactly two qubits.")
        self.n_qubits = n_qubits

        # Define trainable parameters
        self.input_params = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        self.weight_param = Parameter("phi")

        # Build the circuit
        qc = QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        for i, p in enumerate(self.input_params):
            qc.ry(p, i)
        qc.rx(self.weight_param, 0)
        qc.measure_all()

        # Observable: Z⊗Z
        self.observable = Pauli("ZZ")

        # Sampler primitive (use Aer, XACC, or any Qiskit backend)
        self.sampler = Sampler(backend=backend)

        # Instantiate Qiskit EstimatorQNN
        self.estimator = QiskitEstimatorQNN(
            circuit=qc,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=[self.weight_param],
            estimator=self.sampler,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Batch of two‑dimensional input vectors of shape (batch, 2)
            that will be mapped to the Ry angles of the circuit.

        Returns
        -------
        torch.Tensor
            Quantum expectation values as a tensor of shape (batch, 1).
        """
        # Convert to NumPy array for Qiskit
        np_inputs = inputs.detach().cpu().numpy()
        # Predict returns a NumPy array of shape (batch,)
        preds = self.estimator.predict(np_inputs)
        # Convert back to torch tensor
        return torch.tensor(preds, dtype=torch.float32).unsqueeze(1)


__all__ = ["QuantumEstimatorQNN"]
