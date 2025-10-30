"""Hybrid quantum regression model combining torchquantum variational circuit
with a classical head.  The model mirrors the classical architecture but
replaces the random linear layer with a true quantum random layer and uses
a parameterised single‑qubit gate as a lightweight EstimatorQNN analogue.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum import QuantumDevice, QuantumModule, MeasureAll, PauliZ, RandomLayer, RX, RY, GeneralEncoder

def generate_hybrid_data(num_features: int, num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.  The implementation is identical to
    the classical version but uses only numpy for portability.
    """
    # Classical features uniformly sampled in [-1, 1]
    features = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)

    # Map features to angles for quantum state generation
    thetas = np.arccos(features[:, :1].sum(axis=1))
    phis = np.arctan2(features[:, 1:2].sum(axis=1), 1.0)

    # Build superposition states |ψ> = cosθ|0...0> + e^{iφ} sinθ|1...1>
    dim = 2 ** num_wires
    states = np.zeros((samples, dim), dtype=complex)
    for i in range(samples):
        states[i, 0] = np.cos(thetas[i])
        states[i, -1] = np.exp(1j * phis[i]) * np.sin(thetas[i])

    # Target is a smooth function of the angles
    labels = np.sin(2 * thetas) * np.cos(phis)

    return features, states.astype(np.complex64), labels.astype(np.float32)

class HybridRegressionDataset(torch.utils.data.Dataset):
    """Dataset returning classical features, quantum states and labels."""
    def __init__(self, samples: int, num_features: int, num_wires: int):
        self.features, self.states, self.labels = generate_hybrid_data(num_features, num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridQuantumRegression(tq.QuantumModule):
    """Quantum implementation of the hybrid regression model."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            # Quantum random layer (30 ops per wire)
            self.random_layer = RandomLayer(n_ops=30, wires=list(range(num_wires)))
            # Lightweight EstimatorQNN analogue: a single‑qubit RX+RY gate
            self.rx = RX(has_params=True, trainable=True)
            self.ry = RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            # Apply the single‑qubit parametric gate to each wire
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_features: int, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps classical features to a quantum state
        self.encoder = GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = MeasureAll(PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        state_batch: (batch, 2**num_wires) complex tensor
        """
        bsz = state_batch.shape[0]
        qdev = QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["HybridQuantumRegression", "HybridRegressionDataset", "generate_hybrid_data"]
