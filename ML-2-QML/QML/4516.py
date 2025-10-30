from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QLayer(tq.QuantumModule):
    """Variational graph‑like layer with self‑attention style gates."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)
        self.cnot = tq.CNOT(has_params=False)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        for i in range(self.n_wires):
            self.rx(qdev, wires=i)
            self.ry(qdev, wires=i)
            self.rz(qdev, wires=i)
        for i in range(self.n_wires - 1):
            self.crx(qdev, wires=[i, i + 1])
        self.cnot(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

class QFraudLayer(tq.QuantumModule):
    """Quantum analogue of the fraud‑detection photonic layer."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.sx = tq.SX(has_params=True, trainable=True)
        self.cnot = tq.CNOT(has_params=False)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        for i in range(self.n_wires):
            self.rx(qdev, wires=i)
            self.ry(qdev, wires=i)
            self.rz(qdev, wires=i)
        self.sx(qdev, wires=1)
        self.cnot(qdev, wires=[1, 3])
        self.cnot(qdev, wires=[0, 2])

class QuantumNATGen200(tq.QuantumModule):
    """Quantum module mirroring the classical Hybrid model."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layers = [QLayer() for _ in range(3)]
        self.fraud_layer = QFraudLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Encode image: average‑pool to 4 features
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 4)
        self.encoder(qdev, pooled)
        for layer in self.q_layers:
            layer(qdev)
        self.fraud_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QuantumNATGen200"]
