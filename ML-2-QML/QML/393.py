"""Quantum counterpart of the hybrid model: encodes classical features into a 4‑qubit circuit and measures."""
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        """Parameterized quantum layer with random ops, entanglement and single‑qubit rotations."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=60, wires=list(range(n_wires)))
            self.crx = tq.CRX(has_params=True, trainable=True)
            self.cnot01 = tq.CNOT(has_params=False, wires=[0, 1])
            self.cnot23 = tq.CNOT(has_params=False, wires=[2, 3])
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.crx(qdev)
            self.cnot01(qdev)
            self.cnot23(qdev)
            self.rx(qdev)
            self.ry(qdev)
            self.rz(qdev)

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Aggregate spatial dimensions and replicate to match wire count
        pooled = torch.mean(x, dim=(2, 3))          # shape (bsz, 1)
        pooled = pooled.repeat(1, self.n_wires)     # shape (bsz, n_wires)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=True)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
