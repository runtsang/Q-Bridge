"""Hybrid classical regressor that injects quantum‑kernel features into a lightweight MLP."""
from __future__ import annotations

import torch
from torch import nn
import numpy as np
from qiskit import Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

class QuantumFeatureMap(nn.Module):
    """
    Static 3‑qubit feature map that encodes two classical inputs into expectation
    values of Pauli‑Z and Pauli‑Y observables.  The circuit is parameter‑free after
    encoding, making it fast to evaluate on a classical simulator.
    """
    def __init__(self) -> None:
        super().__init__()
        self.backend = Aer.get_backend("statevector_simulator")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(3)
        # Encode the two inputs as Ry rotations on qubits 0 and 1
        qc.ry(Parameter("x0"), 0)
        qc.ry(Parameter("x1"), 1)
        # Simple entangling layer
        qc.cz(0, 2)
        qc.cz(1, 2)
        return qc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor of shape (batch, 2)
            Two classical features per sample.

        Returns
        -------
        torch.Tensor of shape (batch, 6)
            Concatenated expectation values of Z and Y on each qubit.
        """
        batch = inputs.shape[0]
        device = inputs.device
        # Convert to numpy for circuit parameter substitution
        angles = inputs.detach().cpu().numpy()
        z_expect = []
        y_expect = []
        for i in range(batch):
            qc = self.circuit.copy()
            qc.substitute_parameters({qc.params[0]: angles[i, 0], qc.params[1]: angles[i, 1]})
            result = self.backend.run(qc).result()
            sv = Statevector(result.get_statevector(qc))
            z_expect.append(sv.expectation_value('Z'))
            y_expect.append(sv.expectation_value('Y'))
        z_tensor = torch.tensor(z_expect, dtype=inputs.dtype, device=device)
        y_tensor = torch.tensor(y_expect, dtype=inputs.dtype, device=device)
        # Shape (batch, 3) for each observable, concatenate to (batch, 6)
        return torch.cat([z_tensor, y_tensor], dim=1)

class HybridEstimatorQNN(nn.Module):
    """
    Classical regression model that augments a small MLP with quantum‑derived features.
    """
    def __init__(self) -> None:
        super().__init__()
        self.quantum = QuantumFeatureMap()
        self.fc = nn.Sequential(
            nn.Linear(6, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor of shape (batch, 2)

        Returns
        -------
        torch.Tensor of shape (batch,)
            Scalar regression predictions.
        """
        qfeat = self.quantum(inputs)
        out = self.fc(qfeat)
        return out.squeeze(-1)

__all__ = ["HybridEstimatorQNN"]
