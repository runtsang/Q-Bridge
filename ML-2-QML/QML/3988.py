"""Hybrid regression model – quantum implementation.

The class shares the name with the classical version but uses
torchquantum to construct a parameterised quantum circuit.  It
supports a depth‑controlled encoding, an optional random layer,
and a measurement head that feeds into a classical linear layer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate complex‑valued states of the form
    ``cos(theta)|0…0⟩ + e^{i phi} sin(theta)|1…1⟩`` and a corresponding
    regression target.
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


class RegressionDataset(Dataset):
    """Dataset that returns complex state vectors and targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridRegressionModel(tq.QuantumModule):
    """
    Quantum hybrid regression model.

    Architecture:
        * Encoder: GeneralEncoder with Ry rotations per wire.
        * Optional RandomLayer (30 ops) to introduce non‑linearities.
        * Parameterised variational layer: depth‑controlled RX/RZ gates,
          followed by a CZ entangling pattern.
        * Measure all qubits in the Z basis.
        * Classical head: linear layer mapping ``num_wires`` features to a
          scalar output.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, depth: int):
            super().__init__()
            self.n_wires = num_wires
            self.depth = depth
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            # Random layer to break symmetry
            self.random_layer(qdev)
            # Depth‑controlled variational gates
            for _ in range(self.depth):
                for wire in range(self.n_wires):
                    self.rx(qdev, wires=wire)
                    self.rz(qdev, wires=wire)
                for wire in range(self.n_wires - 1):
                    tq.CZ(1.0).apply(qdev, wires=[wire, wire + 1])

    def __init__(self, num_wires: int, depth: int = 2):
        super().__init__()
        self.n_wires = num_wires
        self.depth = depth
        # Encoder uses a pre‑defined Ry‑rotation list per wire
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = self.QLayer(num_wires, depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch: Tensor of shape (batch, 2**num_wires) with dtype=torch.cfloat.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode classical data into the quantum state
        self.encoder(qdev, state_batch)
        # Variational circuit
        self.q_layer(qdev)
        # Extract expectation values
        features = self.measure(qdev)
        # Classical linear head produces the regression output
        return self.head(features).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
