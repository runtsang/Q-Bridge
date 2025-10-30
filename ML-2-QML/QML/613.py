"""Quantum regression model with entangling layers and hybrid classical head."""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum state labels using superposition of |0…0⟩ and |1…1⟩."""
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
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset returning complex state vectors and scalar targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class Entangler(tq.QuantumModule):
    """Entangling layer using CZ gates in a ring topology."""
    def __init__(self, n_wires: int, depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.czs = [tq.CZ(wires=[i, (i + 1) % n_wires]) for i in range(n_wires)]

    def forward(self, qdev: tq.QuantumDevice):
        for _ in range(self.depth):
            for cz in self.czs:
                cz(qdev)


class ParamLayer(tq.QuantumModule):
    """Parameterized rotation layer with trainable RX, RY, and RZ."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
            self.rz(qdev, wires=wire)


class QModel(tq.QuantumModule):
    """Hybrid quantum-classical regression model with entanglement and measurement."""
    def __init__(self, num_wires: int, var_depth: int = 2):
        super().__init__()
        self.n_wires = num_wires
        # State encoder using a parameterized Ry rotation per wire
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Variational block: entanglement + param rotations
        self.entangler = Entangler(num_wires, depth=var_depth)
        self.param_layer = ParamLayer(num_wires)
        # Measurement of expectation values of PauliZ on each wire
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.entangler(qdev)
        self.param_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
