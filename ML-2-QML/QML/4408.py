"""Quantum‑centric counterpart of QuanvolutionHybridNet using TorchQuantum and Qiskit primitives."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.circuit.library import RealAmplitudes
from torchquantum.measurement import MeasureAll, PauliZ
from typing import Tuple

# ----- Quantum components -----

class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Quantum quanvolution filter using 2‑qubit patches."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = MeasureAll(PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuantumSelfAttention(tq.QuantumModule):
    """Quantum self‑attention block."""
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_wires = 784  # 28×28 / 2
        self.rotation = nn.Parameter(torch.randn(self.n_wires))
        self.entangle = nn.Parameter(torch.randn(self.n_wires - 1))
        self.measure = MeasureAll(PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        # Encode each feature as a Ry rotation
        for i in range(self.n_wires):
            qdev.ry(x[:, i], i)
        # Apply learnable rotations
        for i in range(self.n_wires):
            qdev.ry(self.rotation[i], i)
        # Entangle adjacent qubits
        for i in range(self.n_wires - 1):
            qdev.crx(self.entangle[i], i, i + 1)
        measurement = self.measure(qdev)
        # Reshape to (batch, seq_len, embed_dim)
        seq_len = self.n_wires // self.embed_dim
        return measurement.view(bsz, seq_len, self.embed_dim)

class QuantumAutoencoder(tq.QuantumModule):
    """Quantum auto‑encoder that compresses a 784‑dimensional vector to latent_dim."""
    def __init__(self, input_dim: int = 784, latent_dim: int = 32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_wires = input_dim + latent_dim
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(input_dim)]
        )
        self.ansatz = RealAmplitudes(self.n_wires, reps=3)
        self.measure = MeasureAll(PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        self.encoder(qdev, x)
        self.ansatz(qdev)
        measurement = self.measure(qdev)
        # Return measurement of last latent_dim qubits
        return measurement[:, -self.latent_dim:]

class QuantumFullyConnectedLayer(tq.QuantumModule):
    """Quantum fully‑connected layer that maps latent_dim to num_classes."""
    def __init__(self, latent_dim: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.n_wires = latent_dim + num_classes
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(latent_dim)]
        )
        self.ansatz = RealAmplitudes(self.n_wires, reps=3)
        self.measure = MeasureAll(PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        self.encoder(qdev, x)
        self.ansatz(qdev)
        measurement = self.measure(qdev)
        # Return measurement of last num_classes qubits
        return measurement[:, -self.num_classes:]

# ----- Quantum hybrid network -----

class QuanvolutionHybridNet(tq.QuantumModule):
    """
    Quantum‑centric model mirroring the classical hybrid pipeline:
    quanvolution → self‑attention → auto‑encoder → fully‑connected head.
    """
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 10,
                 embed_dim: int = 4,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int,...] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilterQuantum()
        self.attention = QuantumSelfAttention(embed_dim=embed_dim)
        self.autoencoder = QuantumAutoencoder(input_dim=embed_dim * 14 * 14, latent_dim=latent_dim)
        self.fc = QuantumFullyConnectedLayer(latent_dim=latent_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quanvolution
        features = self.qfilter(x)  # (batch, 784)
        # 2. Reshape for attention: (batch, seq_len, embed_dim)
        seq_len = features.shape[1] // self.attention.embed_dim
        features = features.view(features.shape[0], seq_len, self.attention.embed_dim)
        # 3. Self‑attention
        attn = self.attention(features)
        # 4. Flatten back to (batch, 784)
        attn = attn.view(attn.shape[0], -1)
        # 5. Auto‑encoder
        latent = self.autoencoder(attn)
        # 6. Classification head
        logits = self.fc(latent)
        return logits
