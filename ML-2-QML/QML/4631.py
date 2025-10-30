"""Hybrid Self‑Attention model – quantum implementation.

This module reproduces the classical structure using TorchQuantum.
It encodes the input sequence with a parameterised circuit, applies a
quantum kernel via a RandomLayer, measures all qubits and feeds the
feature vector to a linear head.  The class is fully compatible with
the original `SelfAttention.py` API and can be executed on a simulator
or a real back‑end.
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict

class SelfAttentionHybrid(tq.QuantumModule):
    """Quantum hybrid self‑attention + kernel + regression module."""
    def __init__(self, num_wires: int = 4):
        super().__init__()
        self.num_wires = num_wires
        # 1️⃣ Encoder – a simple Ry‑rotation per wire
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        # 2️⃣ Adaptive layer – random entanglement + trainable rotations
        self.qlayer = self.QLayer(num_wires)
        # 3️⃣ Measurement – Pauli‑Z on all wires
        self.measure = tq.MeasureAll(tq.PauliZ)
        # 4️⃣ Classical head
        self.head = nn.Linear(num_wires, 1)

    class QLayer(tq.QuantumModule):
        """Trainable layer applied after the encoder."""
        def __init__(self, num_wires: int):
            super().__init__()
            self.num_wires = num_wires
            self.random = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random(qdev)
            for w in range(self.num_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Shape (batch, seq_len, embed_dim).  The sequence is flattened
            and encoded into a single quantum state per sample.
        """
        bsz = state_batch.shape[0]
        # Flatten the sequence to a 1‑D feature vector per sample
        flat = state_batch.reshape(bsz, -1)
        # 1️⃣ Quantum device for a batch of states
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        # 2️⃣ Encode the classical data
        self.encoder(qdev, flat)
        # 3️⃣ Apply the adaptive trainable layer
        self.qlayer(qdev)
        # 4️⃣ Measurement → feature vector
        features = self.measure(qdev)  # (batch, num_wires)
        # 5️⃣ Linear regression head
        return self.head(features).squeeze(-1)

# ----------------------------------------------------------------------
# Auxiliary data utilities – quantum version
# ----------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a toy regression dataset of superposition states."""
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
    """Dataset wrapper around the quantum superposition data."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

__all__ = ["SelfAttentionHybrid", "RegressionDataset", "generate_superposition_data"]
