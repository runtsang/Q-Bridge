"""Quantum fraud detection circuit using Pennylane.

The circuit encodes a 2‑dimensional feature vector into two qubits,
applies a QCNN‑style block of parameterized two‑qubit gates,
and finishes with a photonic‑inspired squeezing layer and a
measurement that yields a scalar output.  The implementation
uses Pennylane's default `lightning.qubit` device, but can be
easily swapped for a real quantum backend.
"""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn
from typing import Sequence


class FraudDetectionHybrid(nn.Module):
    """
    Quantum model mirroring the classical architecture.  It is
    expressed as a Pennylane QNode wrapped in a PyTorch module so
    that it can be trained with standard optimizers.
    """

    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        # Device can be swapped: "default.qubit", "lightning.qubit", or a real backend
        self.dev = qml.device("lightning.qubit", wires=n_qubits)

        # Parameter register for the QCNN‑style layers
        self.conv_params = nn.Parameter(torch.randn(n_qubits // 2 * 3))
        self.pool_params = nn.Parameter(torch.randn(n_qubits // 2 * 3))
        # Photonic‑inspired squeezing parameters
        self.squeeze_params = nn.Parameter(torch.randn(n_qubits))

        # QNode encapsulating the circuit
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor,
                    conv_p: torch.Tensor,
                    pool_p: torch.Tensor,
                    squeeze_p: torch.Tensor) -> torch.Tensor:
            # Data encoding: two features encoded on qubits 0 and 1
            qml.RY(2 * torch.asin(inputs[0]), wires=0)
            qml.RY(2 * torch.asin(inputs[1]), wires=1)

            # QCNN convolution block
            idx = 0
            for q1, q2 in zip(range(0, n_qubits, 2), range(1, n_qubits, 2)):
                qml.CNOT(q1, q2)
                qml.RZ(conv_p[idx], wires=q1)
                qml.RY(conv_p[idx + 1], wires=q2)
                qml.CNOT(q1, q2)
                idx += 2

            # Pooling block (discard half the qubits by measuring them)
            idx = 0
            for q1, q2 in zip(range(0, n_qubits, 2), range(1, n_qubits, 2)):
                qml.CNOT(q1, q2)
                qml.RZ(pool_p[idx], wires=q1)
                qml.RY(pool_p[idx + 1], wires=q2)
                qml.CNOT(q1, q2)
                idx += 2

            # Photonic‑inspired squeezing on remaining qubits
            for w, r in enumerate(squeeze_p):
                # `qml.Squeezing` is available in Pennylane >=0.28
                qml.Squeezing(r, 0.0, wires=w)

            # Output: expectation of PauliZ on qubit 0
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum circuit.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Output probabilities of shape (batch, 1).
        """
        batch = x.shape[0]
        outputs = []
        for i in range(batch):
            out = self._circuit(
                x[i],
                self.conv_params,
                self.pool_params,
                self.squeeze_params,
            )
            outputs.append(out)
        out = torch.stack(outputs, dim=0)
        return torch.sigmoid(out).unsqueeze(-1)


__all__ = ["FraudDetectionHybrid"]
