"""Hybrid quantum classifier with classical feature extraction.

The quantum part uses a data‑uploading variational ansatz with a
parameterized RandomLayer followed by custom RX/RY/RZ/CRX gates,
mirroring the design of the original Quantum‑NAT QFCModel.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridQuantumClassifier(tq.QuantumModule):
    """
    Quantum counterpart to the classical HybridQuantumClassifier.

    Architecture:
      * 2‑D convolutional backbone (same as classical side).
      * Projection to 4 latent features.
      * General encoder that maps the latent vector onto a 4‑qubit
        device using a 4×4 Ry‑Z‑x‑y gate pattern.
      * A depth‑controlled variational layer consisting of a RandomLayer
        and a small fixed gate sequence.
      * Measurement of all qubits followed by a linear readout.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 4, depth: int = 2):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, num_features: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, num_features))
        self.norm = nn.BatchNorm1d(num_features)

        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires=self.n_wires, depth=depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.readout = nn.Linear(self.n_wires, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        latent = self.fc(flat)
        latent = self.norm(latent)

        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, latent)
        self.q_layer(qdev)
        meas = self.measure(qdev)
        logits = self.readout(meas)
        return logits


__all__ = ["HybridQuantumClassifier"]
