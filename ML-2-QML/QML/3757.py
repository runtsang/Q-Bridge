"""
QuantumEstimator: a lightweight quantum module based on torchquantum.
It implements a parameterized circuit with RX and RY gates followed by a CNOT chain
and returns the expectation of Pauli‑Z on each qubit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumEstimator(tq.QuantumModule):
    """
    Parameterized quantum circuit that acts on ``n_qubits``.
    The circuit consists of:
        - RX and RY gates whose angles are linear functions of the input tensor.
        - A chain of CNOT gates to entangle the qubits.
        - Measurement of Pauli‑Z on each qubit.
    The output is a tensor of shape (batch, n_qubits) containing the expectation values.
    """
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        # Linear map from input to rotation angles
        self.angle_mapper = nn.Linear(n_qubits, n_qubits * 2)  # RX + RY

        # Register quantum circuit components
        self.register_buffer("cnot_pairs", torch.tensor(
            [(i, i + 1) for i in range(n_qubits - 1)], dtype=torch.int32
        ))
        self.register_buffer("pauli_z", torch.tensor([0] * n_qubits, dtype=torch.int32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, n_qubits)
        Returns
        -------
        torch.Tensor
            Shape (batch, n_qubits) with expectation values of Pauli‑Z.
        """
        batch = x.size(0)
        device = x.device

        # Map input to angles
        angles = self.angle_mapper(x)
        rx_angles = angles[:, :self.n_qubits]
        ry_angles = angles[:, self.n_qubits:]

        # Build a quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch, device=device)

        # Apply rotations
        for i in range(self.n_qubits):
            tqf.rx(qdev, wires=i, params=rx_angles[:, i])
            tqf.ry(qdev, wires=i, params=ry_angles[:, i])

        # Entangle with CNOT chain
        for src, dst in self.cnot_pairs:
            tqf.cnot(qdev, wires=[src, dst])

        # Measure expectation of Pauli‑Z
        exp_val = tqf.expval(qdev, wires=range(self.n_qubits), operator="Z")
        return exp_val


__all__ = ["QuantumEstimator"]
