"""Quantum regression dataset and model derived from the seed, extended with
attention over measurement results and a multi‑task output.

The implementation builds on torchquantum and adds an entanglement layer,
an attention module over the Pauli‑Z measurement statistics, and two heads
for regression and binary classification.  This structure enables richer
feature learning while keeping the overall interface identical to the
original seed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from typing import Tuple

def generate_superposition_data(
    num_wires: int,
    samples: int,
    *,
    noise_level: float = 0.0,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.

    The function is a wrapper around the seed implementation that adds
    optional Gaussian noise to the labels.
    """
    rng = np.random.default_rng(random_state)
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    if noise_level > 0.0:
        labels += rng.normal(scale=noise_level, size=labels.shape)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns a binary label indicating high‑frequency content.

    The binary label is 1 if |y| > 0.8 and 0 otherwise.  This encourages the
    model to learn a secondary task that can improve feature learning.
    """
    def __init__(self, samples: int, num_wires: int, noise_level: float = 0.0):
        self.states, self.labels = generate_superposition_data(
            num_wires, samples, noise_level=noise_level
        )
        self.binary = (np.abs(self.labels) > 0.8).astype(np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
            "binary": torch.tensor(self.binary[index], dtype=torch.float32),
        }

class HybridQModel(tq.QuantumModule):
    """Hybrid quantum regression/classification model with attention over
    measurement outcomes.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            # Simple entanglement: a chain of CNOTs
            self.cnot_pairs = [(i, i + 1) for i in range(num_wires - 1)]
            self.cnot = tq.CNOT()
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for control, target in self.cnot_pairs:
                self.cnot(qdev, wires=(control, target))
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps classical amplitudes to the quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Attention over measurement statistics
        self.attn = nn.Linear(num_wires, num_wires)
        # Heads
        self.reg_head = nn.Linear(num_wires, 1)
        self.clf_head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)  # shape (bsz, n_wires)
        attn_weights = torch.softmax(self.attn(features), dim=-1)
        weighted = features * attn_weights
        reg_out = self.reg_head(weighted).squeeze(-1)
        clf_out = torch.sigmoid(self.clf_head(weighted).squeeze(-1))
        return reg_out, clf_out

__all__ = ["HybridQModel", "RegressionDataset", "generate_superposition_data"]
