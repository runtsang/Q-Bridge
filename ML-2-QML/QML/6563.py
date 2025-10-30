"""Quantum variant of QFCModelEnhanced using torchquantum."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QFCModelEnhanced(tq.QuantumModule):
    """Quantum fully‑connected model with 6‑qubit variational layer and 6‑dimensional output."""

    class QLayer(tq.QuantumModule):
        """Variational layer with random gates and parameterised rotations."""
        def __init__(self):
            super().__init__()
            self.n_wires = 6
            self.random_layer = tq.RandomLayer(n_ops=80, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)
            self.cnot0 = tq.CNOT(has_params=False, wires=[0, 1])
            self.cnot1 = tq.CNOT(has_params=False, wires=[2, 3])
            self.cnot2 = tq.CNOT(has_params=False, wires=[4, 5])

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=2)
            self.crx0(qdev, wires=[3, 4])
            self.cnot0(qdev, wires=[0, 1])
            self.cnot1(qdev, wires=[2, 3])
            self.cnot2(qdev, wires=[4, 5])

    def __init__(self):
        super().__init__()
        self.n_wires = 6
        # Encoder that maps classical pooled features onto 6 qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["6x6_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Global average pooling to a 16‑dim vector
        pooled = F.avg_pool2d(x, 6).view(bsz, -1)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QFCModelEnhanced"]
