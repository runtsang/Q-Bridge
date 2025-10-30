"""Hybrid quantum model combining classical CNN extraction with a quantum kernel layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum import op_name_dict, QuantumDevice


class QuantumKernel(tq.QuantumModule):
    """Fixed ansatz that evaluates a quantum kernel (overlap) between two vectors."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = QuantumDevice(n_wires=self.n_wires)
        # Simple ry‑rotation ansatz for each wire
        self.ansatz = [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (batch, 4)
        bsz = x.shape[0]
        self.q_device.reset_states(bsz)
        # encode x
        for info in self.ansatz:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
        # encode y with inverse parameters
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
        # return overlap (magnitude of first state amplitude) for each batch element
        return torch.abs(self.q_device.states[:, 0])


class HybridQuantumNAT(tq.QuantumModule):
    """
    Quantum‑classical hybrid model that mirrors the classical
    HybridQuantumNAT but replaces the RBF kernel with a fixed quantum
    kernel based on a small ry‑rotation ansatz.  The CNN feature
    extractor is kept classical for speed; only the similarity
    computation is quantum, enabling exploration of quantum
    expressivity while remaining trainable on a classical backend.
    """

    def __init__(self, n_prototypes: int = 10) -> None:
        super().__init__()
        # Classical CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Reduce feature dimension to match the quantum device
        self.fc_reduce = nn.Linear(16 * 7 * 7, 4)
        # Quantum kernel module
        self.kernel = QuantumKernel(n_wires=4)
        # Trainable prototypes in the 4‑dimensional reduced space
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, 4))
        # Linear map from kernel similarities to output
        self.fc_out = nn.Linear(n_prototypes, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        reduced = F.relu(self.fc_reduce(flattened))  # (batch, 4)
        # Compute quantum kernel similarities to prototypes
        kernel_sim = torch.stack(
            [self.kernel(reduced, p.expand(bsz, -1)) for p in self.prototypes], dim=1
        )  # (batch, n_prototypes)
        out = self.fc_out(kernel_sim)
        return self.norm(out)


__all__ = ["HybridQuantumNAT"]
