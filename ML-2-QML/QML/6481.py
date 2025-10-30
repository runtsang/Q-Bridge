"""Quantum hybrid model inspired by the original quanvolution and
Quantum‑NAT.

The filter processes the input image in 2×2 patches.  Each patch is
encoded using Ry rotations, then a RandomLayer is applied to mix the
qubits.  A QLayer, mirroring the one in QFCModel, adds trainable
rotations and a controlled‑rotation, followed by a Hadamard, SX and
CNOT.  The measurement yields a 4‑dimensional feature for each patch.
All patch features are concatenated and passed through a BatchNorm1d.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class QuanvolutionHybrid(tq.QuantumModule):
    """Quantum version of the hybrid quanvolution model."""

    class QLayer(tq.QuantumModule):
        """Trainable quantum layer used after the random layer."""
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
            # Trainable rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.crx(qdev, wires=[0, 3])
            # fixed gates
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encoder using Ry on each pixel of the patch
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires * 14 * 14)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        patches = []
        # Reshape to 28x28 per image
        x_img = x.view(bsz, 28, 28)
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x_img[:, r, c],
                        x_img[:, r, c + 1],
                        x_img[:, r + 1, c],
                        x_img[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))
        features = torch.cat(patches, dim=1)
        features = self.norm(features)
        return features


__all__ = ["QuanvolutionHybrid"]
