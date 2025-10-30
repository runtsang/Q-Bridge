"""Quantum regression model with a richer variational ansatz and multi‑observable readout."""
import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate states |ψ(θ,φ)⟩ = cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩ with target y = sin(2θ) cosφ + 0.05 sin(θ)."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis) + 0.05 * np.sin(thetas)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset yielding quantum states and continuous targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumRegressionModel(tq.QuantumModule):
    """Variational circuit with entangling layers and multi‑observable readout."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, depth: int = 3):
            super().__init__()
            self.n_wires = num_wires
            self.depth = depth
            # Parameterized rotation layers
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Entangling layer
            self.entangle = tq.Sequential(
                *[tq.CNOT(wires=[i, (i + 1) % num_wires]) for i in range(num_wires)]
            )

        def forward(self, qdev: tq.QuantumDevice):
            for _ in range(self.depth):
                # Rotations on all wires
                for wire in range(self.n_wires):
                    self.rx(qdev, wires=wire)
                    self.ry(qdev, wires=wire)
                    self.rz(qdev, wires=wire)
                # Entangling pattern
                self.entangle(qdev)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps classical state to quantum amplitudes
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        # Measure all PauliZ and PauliX expectation values
        self.measure_z = tq.MeasureAll(tq.PauliZ)
        self.measure_x = tq.MeasureAll(tq.PauliX)
        # Combine features and feed into a classical head
        self.head = nn.Linear(2 * num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        z = self.measure_z(qdev)
        x = self.measure_x(qdev)
        features = torch.cat([z, x], dim=1)
        return self.head(features).squeeze(-1)

__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
