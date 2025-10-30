"""HybridEstimatorQNN: quantum variational circuit with classical‑style interface.

Features:
  * 2‑D input is encoded via Ry rotations on the first qubit.
  * Trainable rotation (3 per qubit) and entanglement (CRX) gates.
  * Expectation of the average Z observable is returned.
  * Parameters are exposed as buffers for joint optimisation with the classical part.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator

class HybridEstimatorQNN(nn.Module):
    """
    Quantum variational estimator mirroring the classical interface.
    """
    def __init__(self, num_qubits: int = 2) -> None:
        super().__init__()
        self.num_qubits = num_qubits

        # Trainable parameters
        self.rotation_params = ParameterVector("rot", 3 * num_qubits)
        self.entangle_params = ParameterVector("ent", num_qubits - 1)

        # Build the circuit
        self.circuit = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            self.circuit.ry(self.rotation_params[3 * i], i)
            self.circuit.rx(self.rotation_params[3 * i + 1], i)
            self.circuit.rz(self.rotation_params[3 * i + 2], i)
        for i in range(num_qubits - 1):
            self.circuit.crx(self.entangle_params[i], i, i + 1)

        # Observable: average Z over all qubits
        self.observable = SparsePauliOp.from_list([("Z" * num_qubits, 1.0)])

        # Estimator primitive
        self.estimator = Estimator()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: tensor of shape (batch, 2)
        Returns expectation value of the observable for each sample.
        """
        batch = inputs.shape[0]
        results = []
        for idx in range(batch):
            # Map 2‑D input to rotation angles for the first qubit
            params = {
                self.rotation_params[0]: inputs[idx, 0].item(),
                self.rotation_params[1]: inputs[idx, 1].item(),
                self.rotation_params[2]: 0.0,
            }
            # Remaining qubits: zero rotations
            for q in range(1, self.num_qubits):
                params[self.rotation_params[3 * q]] = 0.0
                params[self.rotation_params[3 * q + 1]] = 0.0
                params[self.rotation_params[3 * q + 2]] = 0.0
            # Entanglement parameters set to zero
            for e in range(self.num_qubits - 1):
                params[self.entangle_params[e]] = 0.0

            result = self.estimator.run(
                circuits=self.circuit,
                observables=self.observable,
                parameter_values=[params]
            ).values[0]
            results.append(result)

        return torch.tensor(results, dtype=torch.float32)

__all__ = ["HybridEstimatorQNN"]
