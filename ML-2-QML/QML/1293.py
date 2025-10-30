from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a pair of (states, labels).  The states are a superposition
    of two basis states with random angles, and the labels are
    a trigonometric function of the angles.
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
    """Dataset that returns a dictionary with'states' and 'target' tensors."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ResidualBlock(nn.Module):
    """A lightweight residual block used in the classical head."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return self.bn(x + out)

class ParameterizedAnsatz(tq.QuantumModule):
    """A compact, parameter‑efficient ansatz consisting of
    a few layers of RX/RY/RZ rotations followed by a CNOT chain.
    """
    def __init__(self, num_wires: int, num_layers: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.cnot = tq.CNOT()

    def forward(self, qdev: tq.QuantumDevice) -> None:
        for _ in range(self.num_layers):
            for wire in range(self.num_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
                self.rz(qdev, wires=wire)
            # Entangle neighbouring qubits in a ring topology
            for i in range(self.num_wires):
                self.cnot(qdev, wires=[(i, (i + 1) % self.num_wires)])

class QModel(tq.QuantumModule):
    """Hybrid quantum‑classical regression model with a residual classical head."""
    def __init__(self, num_wires: int, num_layers: int = 2, head_depth: int = 2):
        super().__init__()
        self.n_wires = num_wires
        # Encoder: simple amplitude encoding of the input state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.ansatz = ParameterizedAnsatz(num_wires, num_layers)
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical head: residual MLP
        self.head = nn.Sequential(
            nn.Linear(num_wires, 32),
            nn.ReLU(),
            *[ResidualBlock(32) for _ in range(head_depth)],
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.ansatz(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
