"""Quantum filter module for the QuanvolutionAutoEncoder.

This module provides a patch‑wise 2×2 quantum kernel that can be used as a
submodule in a PyTorch network.  The implementation uses torchquantum
and follows the design of the original quanvolution example.
"""

from __future__ import annotations

import torch
import torchquantum as tq


class QFilter(tq.QuantumModule):
    """Patch‑wise 2×2 quantum kernel applied to every image patch.

    The filter follows the design of the original Quan‑**Q**uantum
    and uses a random variational layer after an encoding circuit.
    """

    def __init__(self, n_wires: int = 4, random_layers: int = 8):
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
        self.q_layer = tq.RandomLayer(n_ops=random_layers, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum kernel to each 2×2 patch of `x`."""
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


__all__ = ["QFilter"]
