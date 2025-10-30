"""Hybrid quantum convolutional network with an explicit quantum expectation head.

This implementation keeps the classical feature extractor from the
original Quantum‑NAT, but replaces the final fully‑connected block
with a parametrised quantum circuit (QLayer).  The circuit is
executed on a quantum device via torchquantum, and the expectation
value of Pauli‑Z is returned as the binary decision probability.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class QLayer(tq.QuantumModule):
    """Parametrised quantum layer used as the quantum head."""

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.rand_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.rand_layer(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=3)
        self.crx(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)


class HybridQFCModel(tq.QuantumModule):
    """Quantum version of the hybrid QFCModel.

    The model keeps the classical CNN backbone and introduces a
    quantum layer that acts as an expectation head.  The output is
    reshaped into a two‑class probability vector.
    """

    def __init__(self) -> None:
        super().__init__()
        # Classical feature extractor (identical to the ML counterpart)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out)
        # Convert the single‑wire expectation to a binary probability
        prob = torch.sigmoid(out[:, 0])
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = ["HybridQFCModel"]
