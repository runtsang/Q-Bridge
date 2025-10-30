"""Quantum regression dataset and hybrid quantum‑kernel model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample quantum states of the form
        cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩
    and a target that depends on θ and φ.
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
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns quantum states and scalar targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class KernalAnsatz(tq.QuantumModule):
    """
    Quantum kernel ansatz that encodes two classical vectors x and y
    via a fixed list of gates and returns the overlap between the two
    encoded states.
    """
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QModel(tq.QuantumModule):
    """
    Hybrid quantum regression model that evaluates a quantum kernel
    against a set of support vectors and feeds the resulting similarity
    vector into a classical linear head.
    """
    def __init__(self, num_wires: int, support_vectors: torch.Tensor, gamma: float = 1.0):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.kernel_ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.support_vectors = support_vectors
        self.gamma = gamma
        self.head = nn.Linear(support_vectors.shape[0], 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        # Encode the input states
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        # Compute quantum‑kernel similarities to each support vector
        features = []
        for sv in self.support_vectors:
            sv_batch = sv.unsqueeze(0).expand(bsz, -1)
            self.kernel_ansatz(qdev, state_batch, sv_batch)
            similarity = torch.abs(qdev.states[:, 0])  # amplitude of |0> for each batch
            features.append(similarity)
        features = torch.stack(features, dim=1)  # (bsz, n_support)
        return self.head(features).squeeze(-1)

__all__ = ["generate_superposition_data", "RegressionDataset", "QModel"]
