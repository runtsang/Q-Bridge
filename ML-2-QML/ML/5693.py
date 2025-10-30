"""Hybrid regression framework with noise‑aware training and multi‑output support."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchquantum as tq
from typing import Optional, Tuple

def generate_superposition_data(
    num_features: int,
    samples: int,
    output_dim: int = 1,
    *,
    noise: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data in the same format as the original seed, but now
    supports multi‑output regression and optional Gaussian noise on
    the labels.
    """
    # Features uniformly in [-1, 1]
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    # Compute angles from feature sums
    angles = x.sum(axis=1)
    # Base labels: sin and cos of angles
    base = np.stack([np.sin(angles), np.cos(angles)], axis=1)  # shape (samples, 2)
    # Random linear combination to produce requested output_dim
    W = np.random.randn(output_dim, 2).astype(np.float32)
    y = base @ W.T  # shape (samples, output_dim)
    if noise > 0.0:
        y += np.random.normal(scale=noise, size=y.shape).astype(np.float32)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that returns a dictionary with "states" and "target".
    """
    def __init__(self, samples: int, num_features: int, output_dim: int = 1):
        self.features, self.labels = generate_superposition_data(num_features, samples, output_dim)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    Hybrid regression model that can operate in classical or quantum mode.
    The quantum mode uses a simple variational circuit with a random layer
    and per‑wire rotations, and optionally adds Gaussian noise to emulate
    decoherence.
    """
    def __init__(
        self,
        num_features: int,
        output_dim: int = 1,
        *,
        use_quantum: bool = False,
        noise_level: float = 0.0,
    ):
        super().__init__()
        self.num_features = num_features
        self.output_dim = output_dim
        self.use_quantum = use_quantum
        self.noise_level = noise_level

        # Classical backbone
        self.classical_net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

        if use_quantum:
            # Quantum encoder: simple Ry rotations per feature
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_features}xRy"])
            # Variational layer
            self.q_layer = _QLayer(num_features)
            # Measurement
            self.measure = tq.MeasureAll(tq.PauliZ)
            # Output head
            self.head = nn.Linear(num_features, output_dim)

    def forward(self, state_batch: torch.Tensor, *, use_quantum: Optional[bool] = None) -> torch.Tensor:
        """
        Forward pass. If `use_quantum` is None, the mode is determined by self.use_quantum.
        """
        quantum = self.use_quantum if use_quantum is None else use_quantum
        if quantum:
            # Quantum forward
            bsz = state_batch.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.num_features, bsz=bsz, device=state_batch.device)
            self.encoder(qdev, state_batch)
            self.q_layer(qdev)
            features = self.measure(qdev)
            if self.noise_level > 0.0:
                noise = torch.randn_like(features) * self.noise_level
                features = features + noise
            return self.head(features).squeeze(-1)
        else:
            # Classical forward
            return self.classical_net(state_batch).squeeze(-1)

class _QLayer(tq.QuantumModule):
    """
    Simple variational layer used in the quantum forward path.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
