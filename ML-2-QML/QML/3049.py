"""Hybrid quantum regression model that prepends a linear preprocessing step
before a variational circuit, mirroring the classical convolutional approach."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Same data generator as the classical side; returns real‑valued features
    and a sinusoidal target. These features are later encoded into a
    quantum state via a linear preprocessing layer.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset mirrors the classical one but provides a 2‑D grid of features
    for the preprocessing layer.
    """
    def __init__(self, samples: int, num_features: int, kernel_size: int = 2):
        self.kernel_size = kernel_size
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.features = self.features.reshape(-1, kernel_size, kernel_size)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridRegression(tq.QuantumModule):
    """Quantum regression model with a linear preprocessing layer."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, kernel_size: int = 2):
        super().__init__()
        self.n_wires = num_wires
        # Linear preprocessing that maps the 2‑D feature grid to a vector
        # of length 2**num_wires, suitable for encoding.
        self.preprocess = nn.Linear(kernel_size * kernel_size, 2 ** num_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # state_batch: (batch, H, W)
        bsz = state_batch.shape[0]
        # flatten and preprocess
        flat = state_batch.view(bsz, -1)  # (batch, H*W)
        encoded = self.preprocess(flat)  # (batch, 2**num_wires)
        # create a quantum device with the encoded states as amplitudes
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=state_batch.device,
            init_state=encoded
        )
        self.encoder(qdev, encoded)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
