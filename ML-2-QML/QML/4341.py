"""Quantum‑centric version of QuanvolutionHybrid.

The quantum model replaces the classical convolution with a variational
quanvolution filter, augments it with a stack of random quantum layers
(acting as a graph‑like quantum neural network), and finally produces a
vector of measurement results that can be fed into a downstream estimator
(e.g. the Qiskit EstimatorQNN).  The API is identical to the classical
counterpart so that the two can be swapped in experiments.
"""

from __future__ import annotations

import torch
import torchquantum as tq
import torch.nn as nn
from typing import List, Sequence

class QuanvolutionHybrid(tq.QuantumModule):
    """Quantum version of the hybrid quanvolution model."""

    def __init__(self,
                 n_wires: int = 4,
                 graph_layers: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires

        # 1. Variational quanvolution filter (from the original QML seed)
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        # 2. Quantum graph‑like layers (stack of random layers)
        self.graph_layers = nn.ModuleList(
            [tq.RandomLayer(n_ops=8, wires=list(range(n_wires))) for _ in range(graph_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum quanvolution filter, graph layers and return measurement."""
        bsz = x.shape[0]
        device = x.device

        # Prepare quantum device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Reshape image to patches
        x = x.view(bsz, 28, 28)
        patches: List[torch.Tensor] = []

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

        # Concatenate all patch measurements
        quanv_features = torch.cat(patches, dim=1)  # shape: (bsz, 4 * 14 * 14)

        # Pass through quantum graph layers
        graph_out = quanv_features
        for layer in self.graph_layers:
            # Re‑encode the current feature vector as Ry rotations
            qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
            for i in range(self.n_wires):
                qdev.apply(tq.RY(), wires=[i], params=graph_out[:, i])
            layer(qdev)
            graph_out = self.measure(qdev).view(bsz, 4)

        return graph_out

__all__ = ["QuanvolutionHybrid"]
