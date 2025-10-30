import torch
import torch.nn as nn
import torch.nn.functional as F
from torchquantum import QuantumDevice, QuantumModule
import torchquantum.functional as tqf
from torchquantum import RandomLayer
from torchquantum import GeneralEncoder
from torchquantum import encoder_op_list_name_dict
from torch.nn.parameter import Parameter

class QuantumNATEnhanced(tq.QuantumModule):
    """
    Quantum variant of the enhanced model.
    Uses a parameter‑shared variational layer after a general encoder.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Parameter‑shared rotation angles
            self.theta = Parameter(torch.randn(3))
            # Random entangling layer
            self.random_layer = RandomLayer(n_ops=20, wires=list(range(n_wires)))
            # Two‑qubit entanglement
            self.entanglement = tqf.cnot

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # Entangle wires with a random layer
            self.random_layer(qdev)
            # Apply shared rotations on each wire
            for w in range(self.n_wires):
                tqf.rx(qdev, self.theta[0], wires=w, static=self.static_mode, parent_graph=self.graph)
                tqf.ry(qdev, self.theta[1], wires=w, static=self.static_mode, parent_graph=self.graph)
                tqf.rz(qdev, self.theta[2], wires=w, static=self.static_mode, parent_graph=self.graph)
            # Add a simple CNOT chain
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = GeneralEncoder(encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)  # 16‑dim encoded vector
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)
