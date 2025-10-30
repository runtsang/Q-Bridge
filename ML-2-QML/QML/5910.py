from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
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
    return torch.tensor(states, dtype=torch.cfloat), torch.tensor(labels, dtype=torch.float32)

class QRegressionDataset(Dataset):
    """
    Dataset that yields complex ``states`` and real ``target`` tensors for training.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"states": self.states[idx], "target": self.labels[idx]}

class QuantumEncoder(tq.QuantumModule):
    """
    Quantum encoder that maps a classical vector into a quantum state.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, x)
        self.q_layer(qdev)
        return self.measure(qdev)

class QuantumFeedForward(tq.QuantumModule):
    """
    Feed‑forward network realised by a quantum module.
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, x)
        for gate in self.params:
            gate(qdev)
        return self.measure(qdev)

class QuantumTransformerBlock(tq.QuantumModule):
    """
    A minimal transformer‑style block that uses quantum circuits for
    both attention and feed‑forward stages.
    """
    def __init__(self, num_wires: int, n_qubits_ffn: int):
        super().__init__()
        self.attn = QuantumEncoder(num_wires)  # placeholder for attention
        self.ffn = QuantumFeedForward(n_qubits_ffn)

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(qdev, x)
        ffn_out = self.ffn(qdev, attn_out)
        return ffn_out

class UnifiedQModel(tq.QuantumModule):
    """
    Hybrid model that stacks a quantum encoder with a classical linear head.
    """
    def __init__(self, num_features: int, num_wires: int):
        super().__init__()
        self.num_features = num_features
        self.num_wires = num_wires
        self.encoder = QuantumEncoder(num_wires)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        enc = self.encoder(qdev, state_batch)
        return self.head(enc).squeeze(-1)

__all__ = [
    "generate_superposition_data",
    "QRegressionDataset",
    "QuantumEncoder",
    "QuantumFeedForward",
    "QuantumTransformerBlock",
    "UnifiedQModel",
]
