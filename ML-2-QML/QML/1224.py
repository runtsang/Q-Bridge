"""Quantum regression model with an enriched variational circuit and kernel utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate states of the form α|0…0⟩ + β|1…1⟩ with random amplitudes.
    The labels are a smooth function of the random angles, mimicking a
    quantum‑inspired regression target.
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
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns complex quantum states and real‑valued targets.
    """
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
    """
    Quantum regression model featuring a deep entangling variational block
    and a helper to compute a quantum kernel matrix.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, depth: int = 4):
            super().__init__()
            self.n_wires = num_wires
            self.depth = depth
            self.layers = nn.ModuleList()
            for _ in range(depth):
                self.layers.append(tq.RandomLayer(n_ops=20, wires=list(range(num_wires))))
                self.layers.append(tq.CNOT(has_params=False, wires=[(i, (i + 1) % num_wires) for i in range(num_wires)]))
                self.layers.append(tq.RX(has_params=True, trainable=True))
                self.layers.append(tq.RZ(has_params=True, trainable=True))

        def forward(self, qdev: tq.QuantumDevice):
            for layer in self.layers:
                layer(qdev)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires, depth=4)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    def predict(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for inference.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(state_batch)

    def compute_kernel(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute a quantum kernel matrix K = Φ(x) Φ(x)ᵀ where Φ(x) are
        expectation values of Pauli‑Z after the variational block.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return torch.mm(features, features.t())


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
