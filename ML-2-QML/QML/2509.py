import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class SelfAttentionHybrid(tq.QuantumModule):
    """
    Quantum‑classical hybrid: quantum self‑attention circuit + quantum feature layer.
    """
    class QSelfAttention(tq.QuantumModule):
        def __init__(self, n_qubits: int = 4):
            super().__init__()
            self.n_qubits = n_qubits
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            for i in range(self.n_qubits):
                self.rx(qdev, wires=i, params=[0.1])  # placeholder params
                self.ry(qdev, wires=i, params=[0.2])
                self.rz(qdev, wires=i, params=[0.3])
            for i in range(self.n_qubits - 1):
                self.crx(qdev, wires=[i, i + 1], params=[0.4])
            tqf.hadamard(qdev, wires=list(range(self.n_qubits)))

    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(4)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=2)
            self.crx0(qdev, wires=[0, 2])

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_attention = self.QSelfAttention(n_qubits=self.n_wires)
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_attention(qdev)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["SelfAttentionHybrid"]
