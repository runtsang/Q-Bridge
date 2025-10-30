"""Quantum hybrid kernel that mirrors the classical extractor and evaluates overlap via amplitude."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumFeatureEncoder(tq.QuantumModule):
    """Parameterised circuit that encodes a 64‑dimensional feature vector."""
    def __init__(self, n_wires: int = 64) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encode each feature onto its own qubit with an Ry rotation
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        # Follow with a random unitary layer to increase expressivity
        self.random = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> None:
        """Apply the encoder and random layer to a batch of data."""
        for i in range(x.shape[0]):
            params = x[i]
            self.encoder(qdev, params)
            self.random(qdev)

class HybridKernelMethod(tq.QuantumModule):
    """Quantum kernel that evaluates the inner product of two encoded states."""
    def __init__(self, n_wires: int = 64) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_dev = tq.QuantumDevice(n_wires=n_wires)
        self.encoder = QuantumFeatureEncoder(n_wires=n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the overlap between two encoded batches."""
        bsz = x.shape[0]
        # Ensure the device handles the batch
        self.q_dev.reset_states(bsz)
        # Encode the first vector
        for i in range(bsz):
            self.encoder(self.q_dev, x[i])
        # Encode the second vector with negated parameters (adjoint)
        for i in range(bsz):
            self.encoder(self.q_dev, -y[i])
        # The amplitude of the all‑zero state equals the inner product
        return torch.abs(self.q_dev.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Utility that builds a Gram matrix for arbitrary tensors using the quantum hybrid kernel."""
    model = HybridKernelMethod()
    return np.array([[model(a_i, b_j).item() for b_j in b] for a_i in a])

__all__ = ["HybridKernelMethod", "kernel_matrix"]
