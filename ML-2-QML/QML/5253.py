"""Hybrid regression model – quantum implementation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq

# --------------------------------------------------------------------- #
# Quantum quanvolution filter
# --------------------------------------------------------------------- #
class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
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
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
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
                self.encoder(self.qdev, data)
                self.q_layer(self.qdev)
                measurement = self.measure(self.qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

# --------------------------------------------------------------------- #
# Hybrid quantum regression model
# --------------------------------------------------------------------- #
class HybridRegressionModel(tq.QuantumModule):
    """Quantum pipeline mirroring the classical counterpart."""
    def __init__(self, num_wires: int = 8) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.embed = nn.Linear(4 * 14 * 14, num_wires)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a batch of images."""
        # Quanvolution
        features = self.qfilter(x)  # (batch, 4*14*14)
        # Embed into qubit space
        embedded = self.embed(features)
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=bsz, device=x.device)
        # Encode classical data
        self.encoder(qdev, embedded)
        # Variational circuit
        self.q_layer(qdev)
        # Measurement
        out = self.measure(qdev)
        # Classical head
        return self.head(out).squeeze(-1)

__all__ = ["HybridRegressionModel"]
