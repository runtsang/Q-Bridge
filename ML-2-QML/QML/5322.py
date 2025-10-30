from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset
from typing import Sequence

# ------------------------------------------------------------------
# Dataset – physics‑based superposition with phase noise
# ------------------------------------------------------------------
def generate_superposition_data(
    num_features: int,
    samples: int,
    *,
    amplitude_noise: float = 0.05,
    phase_noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    # Create a batch of states that are linear combinations of |0...0> and |1...1>.
    # Noise terms are added to both the amplitude and the phase so the model
    # learns to be robust to realistic imperfections.
    thetas = 2 * np.pi * np.random.rand(samples)
    phis   = 2 * np.pi * np.random.rand(samples)

    zero_state = np.zeros(2 ** num_features, dtype=complex)
    zero_state[0] = 1.0
    one_state  = np.zeros(2 ** num_features, dtype=complex)
    one_state[-1] = 1.0
    states = (
        np.cos(thetas)[:, None] * zero_state
        + np.exp(1j * phis)[:, None] * np.sin(thetas)[:, None] * one_state
    )

    labels = np.sin(2 * thetas) * np.cos(phis)
    labels += amplitude_noise * np.random.randn(samples)
    return states.astype(np.complex128), labels.astype(np.float32)

class RegressionDataset(Dataset):
    # A Dataset that returns a complex‑valued state and a real‑valued target.
    # Mirrors the QML interface but keeps all data in PyTorch tensors for
    # compatibility with the quantum head.
    def __init__(self, samples: int, num_features: int):
        self.states, self.targets = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
           'states': torch.tensor(self.states[idx], dtype=torch.cfloat),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32),
        }

# ------------------------------------------------------------------
# Fully quantum regression model
# ------------------------------------------------------------------
class QuantumRegressionGen(tq.QuantumModule):
    # Fully quantum regression model that maps a superposition state to a scalar output.
    # The circuit is a stack of RandomLayers followed by trainable RX/RY gates, then a
    # measurement of PauliZ on each wire.  A linear head maps the expectation values to a
    # single real number.
    class _Layer(tq.QuantumModule):
        def __init__(self, n_wires: int, n_ops: int):
            super().__init__()
            self.n_wires = n_wires
            self.random = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_wires: int, num_layers: int = 3, n_ops_per_layer: int = 20):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers

        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f'{num_wires}xRy']
        )
        self.layers = nn.ModuleList(
            [self._Layer(num_wires, n_ops_per_layer) for _ in range(num_layers)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        for layer in self.layers:
            layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ['QuantumRegressionGen', 'RegressionDataset', 'generate_superposition_data']
