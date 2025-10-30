"""Hybrid quantum kernel model using TorchQuantum, combining encoding, random layers, and scaling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

class HybridKernelAnsatz(tq.QuantumModule):
    """Quantum ansatz that encodes data with Ry gates, adds a random layer, and applies trainable RX/RY."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        self.encoder(q_device, x)
        self.random_layer(q_device)
        for wire in range(self.num_wires):
            self.rx(q_device, wires=wire)
            self.ry(q_device, wires=wire)

    def get_features(self, q_device: tq.QuantumDevice) -> torch.Tensor:
        features = self.measure(q_device)
        return self.scale * features + self.shift

class HybridKernelModel(tq.QuantumModule):
    """Quantum kernel model with a linear head."""
    def __init__(self, num_wires: int, out_dim: int = 1):
        super().__init__()
        self.num_wires = num_wires
        self.ansatz = HybridKernelAnsatz(num_wires)
        self.head = nn.Linear(num_wires, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=x.device)
        self.ansatz(qdev, x)
        features = self.ansatz.get_features(qdev)
        return self.head(features).squeeze(-1)

def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor], num_wires: int) -> np.ndarray:
    """Compute Gram matrix by evaluating the quantum kernel."""
    model = HybridKernelModel(num_wires)
    return np.array([[model(x.unsqueeze(0), y.unsqueeze(0)).item() for y in b] for x in a])

__all__ = ["HybridKernelModel", "kernel_matrix"]
