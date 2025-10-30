"""Hybrid quantum‑classical model that merges a convolutional front‑end with a variational circuit."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter


class FCLQuantumHybrid(nn.Module):
    """
    Quantum‑classical hybrid architecture.

    1. Convolutional feature extractor (same as the classical variant).
    2. Fully‑connected projection to 4 parameters.
    3. Variational circuit that maps each parameter to a Ry gate on a qubit
       and measures Z‑expectation values.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits in the circuit.
    backend : qiskit.providers.backend.Backend, optional
        Backend to execute the circuit.  Defaults to Aer qasm simulator.
    shots : int, default 1024
        Number of shots for expectation estimation.
    """

    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Convolutional front‑end
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_qubits),
        )

        # Prepare a parameterised circuit template
        self.theta = [Parameter(f"θ_{i}") for i in range(n_qubits)]
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        for i in range(n_qubits):
            self.circuit.ry(self.theta[i], i)
        self.circuit.measure_all()

    def _expectation(self, pauli_string: str, counts: dict) -> float:
        """Compute expectation of a Pauli string given measurement counts."""
        exp = 0.0
        total = sum(counts.values())
        for bitstring, cnt in counts.items():
            prob = cnt / total
            parity = 1
            for i, p in enumerate(pauli_string[::-1]):  # LSB first
                if p == "Z":
                    parity *= -1 if bitstring[i] == "1" else 1
            exp += parity * prob
        return exp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Quantum expectation values of shape (batch, n_qubits).
        """
        # Feature extraction
        out = self.features(x)
        out = out.view(out.size(0), -1)

        # Linear projection to parameters
        params = self.fc(out)  # shape (batch, n_qubits)

        batch_expectations = []
        for i in range(params.size(0)):
            param_bind = {self.theta[j]: float(params[i, j].item()) for j in range(self.n_qubits)}
            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[param_bind])
            result = job.result()
            counts = result.get_counts(self.circuit)

            # Expectation for each qubit (Z basis)
            exp_vals = []
            for q in range(self.n_qubits):
                pauli = "Z" * q + "I" * (self.n_qubits - q - 1)
                exp_vals.append(self._expectation(pauli, counts))
            batch_expectations.append(exp_vals)

        return torch.tensor(batch_expectations, dtype=torch.float32)


__all__ = ["FCLQuantumHybrid"]
