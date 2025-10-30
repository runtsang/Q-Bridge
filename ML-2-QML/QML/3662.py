"""Quantum regression model with a variational self‑attention layer.

The architecture mirrors the classical model but replaces the feed‑forward
head with a quantum variational circuit.  A dedicated ``QLayer`` implements
a quantum self‑attention style block: trainable rotation angles drive
RX‑RY‑RZ gates on each qubit, followed by a chain of controlled‑RX
entangling operations.  The overall model therefore blends classical
feature encoding, quantum self‑attention, and a linear readout.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum import QuantumModule

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate complex‑valued states and non‑linear targets.

    Parameters
    ----------
    num_wires : int
        Number of qubits in each state.
    samples : int
        Number of samples to generate.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Complex state matrix (samples × 2ⁿ) and target vector.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapper compatible with the quantum forward pass."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QLayer(QuantumModule):
    """Variational layer implementing a quantum self‑attention block."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires

        # Random rotation layer to initialise entanglement structure.
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))

        # Trainable rotation angles for RX‑RY‑RZ on each qubit.
        self.rot_params = nn.Parameter(torch.randn(num_wires, 3))

        # Trainable controlled‑RX angles for neighbour entanglement.
        self.entangle_params = nn.Parameter(torch.randn(num_wires - 1))

        # Quantum gates with trainable parameters.
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        # Apply per‑qubit rotations driven by the self‑attention parameters.
        for i in range(self.n_wires):
            self.rx(qdev, theta=self.rot_params[i, 0], wires=i)
            self.ry(qdev, theta=self.rot_params[i, 1], wires=i)
            self.rz(qdev, theta=self.rot_params[i, 2], wires=i)
        # Entangle neighbours with controlled‑RX gates.
        for i in range(self.n_wires - 1):
            self.crx(qdev, theta=self.entangle_params[i], wires=[i, i + 1])

class QModel(QuantumModule):
    """Quantum regression model using a classical encoder, a variational
    self‑attention block, and a linear readout.
    """

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires

        # Classical encoding of the input state into rotation angles.
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )

        # Variational self‑attention block.
        self.q_layer = QLayer(num_wires)

        # Measurement of all qubits in the Pauli‑Z basis.
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical linear head mapping measurement outcomes to a real value.
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Complex state vector batch of shape (batch, 2ⁿ).

        Returns
        -------
        torch.Tensor
            Predicted scalar values of shape (batch,).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)

        # Encode the classical data onto the quantum device.
        self.encoder(qdev, state_batch)

        # Apply the variational self‑attention circuit.
        self.q_layer(qdev)

        # Extract expectation values of Pauli‑Z on each qubit.
        features = self.measure(qdev)

        # Linear readout.
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
