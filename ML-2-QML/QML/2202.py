"""Hybrid Quanvolution model – quantum‑classical implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuanvolutionHybridQML(tq.QuantumModule):
    """
    Quantum‑classical hybrid that mirrors the classical architecture but
    replaces the final feature extraction with a quantum kernel applied to
    2×2 image patches.  It uses a random layer, parametric gates, and a
    measurement‑based readout, followed by a linear classifier.
    """

    class QLayer(tq.QuantumModule):
        """Parametric quantum layer used for each patch."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder that maps a 4‑dim patch into the 4‑qubit Hilbert space
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
        self.norm = nn.BatchNorm1d(self.n_wires)

        # Classical head – identical to the ML version
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Classical feature extraction
        features = nn.functional.max_pool2d(
            nn.functional.relu(nn.functional.conv2d(x, torch.ones(8, 1, 3, 3))),
            kernel_size=2,
        )
        # Quantum kernel applied to 2×2 patches
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device, record_op=True)

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
        quantum_features = torch.cat(patches, dim=1)
        quantum_features = self.norm(quantum_features)

        # Combine with classical features
        flattened = features.view(bsz, -1)
        combined = torch.cat([flattened, quantum_features], dim=1)

        logits = self.classifier(combined)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybridQML"]
