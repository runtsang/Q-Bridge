"""Quantum hybrid regression model using torchquantum and a sampler circuit."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.circuit import QuantumCircuit


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>
    and labels based on a non‑linear function of theta and phi.
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


class QuantumRegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns quantum state vectors and the associated target.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return self.states.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class SamplerQNN(tq.QuantumModule):
    """
    Quantum sampler circuit producing a two‑outcome probability distribution.
    Mirrors the classical SamplerModule but implemented with a parameterised
    circuit and a StatevectorSampler.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.sampler = tq.StatevectorSampler()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        for w in range(self.num_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)
        probs = self.sampler(qdev)
        return probs  # shape (batch, 2)


class QuantumHybridRegression(tq.QuantumModule):
    """
    Combines a quantum encoder, a sampler QNN, and a classical head to produce
    a regression output. The sampler provides a 2‑dimensional probability vector
    that is concatenated with expectation values from the encoder.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.sampler_qnn = SamplerQNN(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires + 2, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        exp_vals = self.measure(qdev)  # expectation values of Pauli‑Z on each wire
        probs = self.sampler_qnn(qdev)
        features = torch.cat([exp_vals, probs], dim=-1)
        return self.head(features).squeeze(-1)


__all__ = ["QuantumHybridRegression", "QuantumRegressionDataset", "SamplerQNN", "generate_superposition_data"]
