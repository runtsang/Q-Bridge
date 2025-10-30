"""Quantum regression model with an entangling variational ansatz.

This module extends the seed circuit by adding a dedicated
entanglement layer and a stack of parameterised rotations.
Measurement results are fed into a hybrid classical head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Create superposition states with tunable phases and angles."""
    omega_0 = np.zeros(2**num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2**num_wires, dtype=complex)
    omega_1[-1] = 1.0

    rng = np.random.default_rng(seed=42)
    thetas = rng.uniform(0, 2 * np.pi, size=samples)
    phis = rng.uniform(0, 2 * np.pi, size=samples)
    states = np.array(
        [np.cos(theta) * omega_0 + np.exp(1j * phi) * np.sin(theta) * omega_1
         for theta, phi in zip(thetas, phis)],
        dtype=complex,
    )
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that stores quantum states and regression targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(tq.QuantumModule):
    """Hybrid quantum‑classical regression circuit."""

    class EntanglementLayer(tq.QuantumModule):
        """Entangles neighbouring wires with CNOT gates."""

        def __init__(self, num_wires: int):
            super().__init__()
            self.num_wires = num_wires
            self.cnot = tq.CNOT(has_params=False, trainable=False)

        def forward(self, qdev: tq.QuantumDevice):
            for i in range(self.num_wires - 1):
                self.cnot(qdev, wires=[i, i + 1])
            # Wrap around to create a ring topology
            self.cnot(qdev, wires=[self.num_wires - 1, 0])

    class VariationalLayer(tq.QuantumModule):
        """Stack of parameterised rotations with trainable angles."""

        def __init__(self, num_wires: int, n_layers: int = 3):
            super().__init__()
            self.num_wires = num_wires
            self.n_layers = n_layers
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            for _ in range(self.n_layers):
                for w in range(self.num_wires):
                    self.rx(qdev, wires=w)
                    self.ry(qdev, wires=w)
                    self.rz(qdev, wires=w)

    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        # State‑encoding using a simple Ry rotation per wire
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.entangle = self.EntanglementLayer(num_wires)
        self.var = self.VariationalLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head with a small MLP
        self.head = nn.Sequential(
            nn.Linear(num_wires, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Execute the circuit on a batch of quantum states."""
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.entangle(qdev)
        self.var(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
