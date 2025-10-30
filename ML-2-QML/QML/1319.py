"""Enhanced quantum model with parameter‑shared ansatz and circuit‑cutting."""
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QFCModelEnhanced(tq.QuantumModule):
    """Quantum version of the enhanced model."""
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
            self.rx = tq.RX
            self.rz = tq.RZ
            self.cnot = tq.CNOT
            self.n_layers = 2

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for _ in range(self.n_layers):
                for i in range(self.n_wires):
                    self.rx(qdev, wires=i)
                    self.rz(qdev, wires=i)
                for i in range(self.n_wires):
                    self.cnot(qdev, wires=[i, (i+1)%self.n_wires])

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Encode input: average pooling to 16 features
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QFCModelEnhanced"]
