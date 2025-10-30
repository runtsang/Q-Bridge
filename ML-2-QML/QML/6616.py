"""Quantum hybrid kernel combining quanvolution filter and quantum kernel."""
from __future__ import annotations

import numpy as np
from typing import Sequence
import torch
import torchquantum as tq

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Apply a random quantum layer to 2x2 patches."""
    def __init__(self, n_wires: int = 4, n_ops: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class HybridKernel(tq.QuantumModule):
    """Quantum kernel that first maps data through quanvolution filter then computes overlap."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.filter = QuantumQuanvolutionFilter(n_wires=self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        fx = self.filter(x)
        fy = self.filter(y)
        # Compute normalized inner product of the resulting feature vectors
        dot = torch.dot(fx.squeeze(), fy.squeeze())
        norm = torch.norm(fx) * torch.norm(fy)
        return dot / (norm + 1e-8)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = HybridKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridKernel", "kernel_matrix"]
