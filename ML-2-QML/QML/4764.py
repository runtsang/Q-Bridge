"""Quantum hybrid regression model mirroring the classical `HybridRegression` API.

Key extensions
---------------
* Uses a `GeneralEncoder` to map classical features to a quantum state.
* Includes a lightweight random variational layer (optional).
* A dedicated `SamplerQNNQuantum` module that turns a 2‑dimensional vector into a probability
  distribution, mirroring the classical sampler.
* The architecture is fully compatible with the classical version, enabling side‑by‑side
  experiments.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate superposition states |ψ⟩ = cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩."""
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
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class SamplerQNNQuantum(tq.QuantumModule):
    """A tiny quantum sampler that maps a 2‑qubit circuit to a probability distribution."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 2
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["2xRy"])
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.rx = tq.RX(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(self.n_wires, 2)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # ensure complex dtype
        state_batch = state_batch.to(torch.cfloat)
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        for wire in range(self.n_wires):
            self.rz(qdev, wires=wire)
            self.rx(qdev, wires=wire)
        features = self.measure(qdev)
        return F.softmax(self.head(features), dim=-1)


class HybridRegression(tq.QuantumModule):
    """
    Quantum hybrid regression that encodes the input, optionally applies a random
    variational layer, and emits a 2‑dimensional probability distribution.
    """

    def __init__(self, num_wires: int, use_random_layer: bool = True) -> None:
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.use_random_layer = use_random_layer
        if self.use_random_layer:
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.head = nn.Linear(num_wires, 2)
        self.sampler = SamplerQNNQuantum()

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        if self.use_random_layer:
            self.random_layer(qdev)
        features = tq.MeasureAll(tq.PauliZ)(qdev)
        logits = self.head(features)
        probs = self.sampler(logits)
        return probs  # return probability distribution


__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data", "SamplerQNNQuantum"]
