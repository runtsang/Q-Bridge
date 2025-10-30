"""Quantum quanvolutional regressor with a variational read‑out.

The filter encodes 2×2 image patches into a 4‑qubit state, applies a random circuit
plus trainable RX/RY rotations, and measures all wires.  The resulting
feature vector is linearly mapped to a scalar target, mirroring the regression
head of the QuantumRegression example.  This design allows a direct
comparison of classical and quantum scaling for the same task.
"""

import torch
import torch.nn as nn
import torchquantum as tq


class QuanvolutionFilter(tq.QuantumModule):
    """Quantum filter that processes 2×2 image patches with a random circuit
    and trainable RX/RY layers.
    """
    def __init__(self, n_wires: int = 4, n_random_ops: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encode each pixel into a single qubit using Ry rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=n_random_ops, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
                self.random_layer(qdev)
                for wire in range(self.n_wires):
                    self.rx(qdev, wires=wire)
                    self.ry(qdev, wires=wire)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))
                qdev.reset()  # reset for next patch
        return torch.cat(patches, dim=1)


class QuanvolutionRegressor(tq.QuantumModule):
    """Quantum regression network built from the quanvolution filter."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.qfilter = QuanvolutionFilter(n_wires=n_wires)
        self.head = nn.Linear(n_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)  # shape (B, 4*14*14)
        patch_count = features.shape[1] // self.n_wires
        # Aggregate patches by averaging before linear read‑out
        features = features.view(x.size(0), patch_count, self.n_wires).mean(dim=1)
        return self.head(features).squeeze(-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionRegressor"]
