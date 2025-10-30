"""Quantum regression model with adjustable entangling depth and encoder choice."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum superposition states and labels."""
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
    """Dataset returning complex quantum states and regression targets."""
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
    """Variational quantum regression model with configurable depth and encoder."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, depth: int = 2):
            super().__init__()
            self.n_wires = num_wires
            self.depth = depth
            self.layers = nn.ModuleList()
            for _ in range(depth):
                # Random parameterized layer
                self.layers.append(tq.RandomLayer(n_ops=15, wires=list(range(num_wires))))
                # Entangling pattern
                self.layers.append(tq.CNOT(control=0, target=1))
                self.layers.append(tq.CNOT(control=1, target=0))
                # Single‑qubit rotations
                self.layers.append(tq.RX(has_params=True, trainable=True))
                self.layers.append(tq.RY(has_params=True, trainable=True))
                self.layers.append(tq.RZ(has_params=True, trainable=True))

        def forward(self, qdev: tq.QuantumDevice):
            for layer in self.layers:
                layer(qdev)

    def __init__(self, num_wires: int, encoder: str = "Ry", depth: int = 2):
        super().__init__()
        self.n_wires = num_wires
        self.encoder_name = encoder
        self.depth = depth
        # Encoder can be "Rx", "Ry", or "Rz"
        if encoder in {"Rx", "Ry", "Rz"}:
            self.encoder = tq.GeneralEncoder([tq.encoder_op_list_name_dict[f"{num_wires}x{encoder}"]])
        else:
            # Default to a generic Ry encoder if an unknown name is provided
            self.encoder = tq.GeneralEncoder([tq.encoder_op_list_name_dict[f"{num_wires}xRy"]])
        self.q_layer = self.QLayer(num_wires, depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(X)

    def evaluate(self, X: torch.Tensor, y_true: torch.Tensor) -> dict[str, float]:
        preds = self.predict(X)
        mse = ((preds - y_true) ** 2).mean().item()
        mae = torch.abs(preds - y_true).mean().item()
        return {"mse": mse, "mae": mae}


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
