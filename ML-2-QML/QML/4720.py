"""Quantum regression module using Pennylane.

The quantum model follows the structure of the QML seed:
* a rotation‑encoding of the classical input,
* a trainable variational layer,
* measurement of Pauli‑Z on each qubit,
* a classical linear head that maps the measurement vector to a scalar.

The implementation is intentionally lightweight and fully compatible with
CPU simulation.  It can be executed in a Jupyter notebook or integrated
into a larger training pipeline.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Data generation (identical to the classical seed for consistency)
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_wires: int, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data that follows
    ``sin(2θ) * cos(φ)`` where θ and φ are random angles.
    """
    rng = np.random.default_rng()
    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        omega0 = np.zeros(2 ** num_wires, dtype=complex)
        omega0[0] = 1.0
        omega1 = np.zeros(2 ** num_wires, dtype=complex)
        omega1[-1] = 1.0
        states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

# --------------------------------------------------------------------------- #
# Quantum model
# --------------------------------------------------------------------------- #
class HybridRegression(nn.Module):
    """
    Quantum regression model that encodes a classical vector into a
    quantum state, processes it with a trainable variational layer,
    measures Pauli‑Z on each qubit, and applies a linear head.

    Parameters
    ----------
    num_wires : int
        Number of qubits used for encoding and computation.
    """
    def __init__(self, num_wires: int, device: str = "default.qubit"):
        super().__init__()
        self.num_wires = num_wires
        self.dev = qml.device(device, wires=num_wires, shots=None)
        # trainable parameters for the variational layer
        self.theta = nn.Parameter(torch.randn(num_wires))
        # classical linear head
        self.head = nn.Linear(num_wires, 1)

        # build the QNode
        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            # Angle encoding
            for i in range(num_wires):
                qml.RY(x[i], wires=i)
            # Variational layer
            for i in range(num_wires):
                qml.RZ(self.theta[i], wires=i)
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch, num_wires)``.  Each row is the classical
            data to be encoded into the quantum circuit.

        Returns
        -------
        torch.Tensor
            Regression output of shape ``(batch,)``.
        """
        # Ensure input is on the same device as the parameters
        x = x.to(self.theta.device)
        # Quantum expectation values
        z = self.circuit(x)
        z = torch.stack(z, dim=1)  # shape: (batch, num_wires)
        # Classical head
        out = self.head(z).squeeze(-1)
        return out

__all__ = ["HybridRegression", "generate_superposition_data"]
