import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATHybrid(tq.QuantumModule):
    """
    Quantum branch of the hybrid QuantumNAT model.
    Implements a deeper parameterised ansatz with multiple rotation layers
    and entangling gates, followed by a measurement and a small
    classical post‑processing network.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, n_layers: int = 3):
            super().__init__()
            self.n_wires = n_wires
            self.n_layers = n_layers
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            for _ in range(self.n_layers):
                for w in range(self.n_wires):
                    self.rx(qdev, wires=w)
                    self.ry(qdev, wires=w)
                    self.rz(qdev, wires=w)
                # Entangle adjacent qubits in a ring
                for w in range(self.n_wires):
                    tqf.cnot(qdev, wires=[w, (w + 1) % self.n_wires])
            return qdev

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(self.n_wires, n_layers=3)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.post_fc = nn.Sequential(
            nn.Linear(self.n_wires, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Pool input to match encoder dimension
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, -1)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.post_fc(out)
        return self.norm(out)
