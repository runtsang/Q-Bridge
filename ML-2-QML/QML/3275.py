"""Quantum regression model that augments a variational circuit with
a parameterised fully‑connected layer (Qiskit or TorchQuantum).
The network is fully differentiable when using the TorchQuantum
backend and can be executed on real quantum hardware via Qiskit.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from typing import Iterable

# --------------------------------------------------------------------------- #
# Data generation utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum states in the form
    cos(theta)|0..0> + exp(i*phi) sin(theta)|1..1>.
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
    """Dataset yielding quantum states and scalar targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Quantum‑style fully‑connected layer
# --------------------------------------------------------------------------- #
class QFullyConnectedLayer(tq.QuantumModule):
    """A minimal parameterised quantum circuit that mimics the behaviour
    of the classical FCL but operates on a single qubit.  The circuit
    consists of a rotation around X and Y, followed by a measurement
    in the Z basis.  The expectation value is used as a feature.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        # Random layer to increase expressivity
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
        # Parameterised rotations
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class QModel(tq.QuantumModule):
    """Quantum regression model that combines a variational encoder,
    a parameterised quantum layer, and a classical read‑out head.

    The encoder transforms input amplitudes into a quantum state.
    The QFullyConnectedLayer injects non‑linear quantum features.
    The final measurement is followed by a linear head that maps
    the expectation values to a scalar target.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encode classical data into a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Parameterised quantum block
        self.q_layer = QFullyConnectedLayer(num_wires)
        # Measure all qubits in Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical read‑out head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for a batch of input states.

        Parameters
        ----------
        state_batch : torch.Tensor
            Tensor of shape (batch, 2**num_wires) with complex dtype.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode
        self.encoder(qdev, state_batch)
        # Quantum feature block
        self.q_layer(qdev)
        # Expectation values
        features = self.measure(qdev)
        # Classical head
        return self.head(features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
