"""QuantumRegressionHybrid: quantum regression model.

This module implements a hybrid quantum‑classical network that can
process either real‑valued feature vectors or full complex state
vectors.  The design borrows the encoder and random layer from the
classical seed, and the measurement head from the quantum seed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int,
                                samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate complex state vectors and labels, identical to the
    classical seed but expressed in terms of qubit count.
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

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns complex state vectors and continuous targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """
    Hybrid quantum regression module.  The encoder maps real‑valued
    features into a quantum state; if the input is already a complex
    vector the encoder is bypassed.  A random layer followed by
    parameterised RX/RY gates produces a feature vector that is
    collapsed with a linear head.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that accepts real‑valued input and maps it onto qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Either a real‑valued tensor of shape (B, N) or a complex
            tensor of shape (B, 2**N).  The method automatically
            detects the dtype and routes the data through the
            encoder or directly to the quantum device.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)

        if torch.is_floating_point(state_batch):
            # Real‑valued features → encode
            self.encoder(qdev, state_batch)
        else:
            # Complex vector → already a state; directly load
            qdev.state = state_batch.to(torch.cfloat)

        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["RegressionDataset", "QModel", "generate_superposition_data"]
