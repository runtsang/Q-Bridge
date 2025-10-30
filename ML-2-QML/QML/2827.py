"""Hybrid quantum regression model that fuses a classical convolutional encoder with a variational quantum layer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate superposition states |ψ(θ,φ)⟩ = cosθ |0…0⟩ + e^{iφ} sinθ |1…1⟩."""
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
    """Dataset that returns quantum states and continuous targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """
    Quantum regression model that mirrors the classical architecture:
      * A lightweight classical CNN (on a 1‑D representation of the state amplitudes)
      * A GeneralEncoder that maps the processed features to qubit amplitudes
      * A variational QLayer that applies random gates followed by trainable RX/RY rotations
      * Measurement of all qubits and a linear read‑out head.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            # Random layer injects expressivity
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            # Trainable single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_wires: int = 4, conv_channels: int = 8):
        super().__init__()
        self.n_wires = num_wires
        # Classical encoder mirroring the CNN in the classical model
        self.features = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # General encoder that maps classical features to qubit amplitudes
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        self.norm = nn.BatchNorm1d(1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Shape (B, 2^N) complex amplitudes of the quantum state.
        Returns
        -------
        torch.Tensor
            Shape (B,) regression predictions.
        """
        bsz = state_batch.shape[0]
        # Convert to a 1‑D image for the classical feature extractor
        x = state_batch.abs().unsqueeze(1)  # use magnitude as 1‑D signal
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        # Encode the classical features into the quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device, record_op=True)
        self.encoder(qdev, flat)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(self.head(out).squeeze(-1))

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
