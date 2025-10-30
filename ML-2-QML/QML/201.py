"""Quantum variant of the extended model using a variational circuit and measurement."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QFCModelGen222(tq.QuantumModule):
    """Quantum fully connected model with enhanced variational block."""
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.static_mode = True  # enable static gates
            # More expressive encoder: parameterized rotation layers
            self.encoder = tq.RandomLayer(n_ops=70, wires=list(range(self.n_wires)),
                                          has_params=True)
            self.entangle = tq.CNOT(wires=[0, 1], has_params=False)
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # Encoder
            self.encoder(qdev)
            # Entanglement layer
            self.entangle(qdev)
            # Parameterized rotations
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.crx(qdev, wires=[0, 3])
            # Additional static gates
            tqf.hadamard(qdev, wires=3, static=self.static_mode)
            tqf.sx(qdev, wires=2, static=self.static_mode)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Use a more expressive general encoder mapping 4x4 images to qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=True)
        # Use adaptive pooling to match 4x4 spatial size
        pooled = F.adaptive_avg_pool2d(x, output_size=(4, 4)).view(bsz, -1)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["QFCModelGen222"]
