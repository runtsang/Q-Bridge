"""Quantum‑enabled version of QuantumNATGen using torchquantum."""

from __future__ import annotations

import torch
from torch import nn
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumNATGen(tq.QuantumModule):
    """
    Hybrid quantum–classical model.
    * Classical CNN extracts features from 28×28 images.
    * Features are encoded into 4 qubits via a 4‑qubit GeneralEncoder.
    * A variational QLayer with random and Euler gates entangles the qubits.
    * Measurement of all qubits is interpreted as a 4‑dimensional vector.
    * A classical linear head maps the quantum expectation to final outputs.
    """

    class QLayer(tq.QuantumModule):
        """Parameterised variational circuit used after encoding."""

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
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

        # Classical head to map 4‑dimensional quantum output to final 4‑dimensional regression
        self.classical_head = nn.Sequential(
            nn.Linear(self.n_wires, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Classical feature extraction
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)

        # Variational entanglement
        self.q_layer(qdev)

        # Quantum measurement
        q_out = self.measure(qdev)          # (bsz, 4)
        q_out = self.norm(q_out)

        # Classical head
        out = self.classical_head(q_out)
        return out


__all__ = ["QuantumNATGen"]
