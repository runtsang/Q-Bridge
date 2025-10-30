import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum variant with a depth‑controlled variational encoder and scaling."""
    class VariationalEncoder(tq.QuantumModule):
        def __init__(self, n_wires: int, depth: int):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            self.layers = nn.ModuleList()
            for _ in range(depth):
                layer = tq.QuantumModule()
                layer.rx = tq.RX(has_params=True, trainable=True)
                layer.ry = tq.RY(has_params=True, trainable=True)
                layer.rz = tq.RZ(has_params=True, trainable=True)
                layer.cnot = tq.CNOT
                self.layers.append(layer)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            for layer in self.layers:
                for w in range(self.n_wires):
                    layer.rx(qdev, wires=w)
                for w in range(self.n_wires):
                    layer.ry(qdev, wires=w)
                for w in range(self.n_wires):
                    layer.rz(qdev, wires=w)
                for i in range(0, self.n_wires - 1, 2):
                    layer.cnot(qdev, wires=[i, i + 1])

    def __init__(self, depth: int = 3):
        super().__init__()
        self.n_wires = 4
        self.encoder = self.VariationalEncoder(self.n_wires, depth)
        self.q_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.scaling = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Encode classical features into the quantum state
        for i in range(16):
            tqf.rx(qdev, params=pooled[:, i], wires=i % self.n_wires)
        self.encoder(qdev)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out) * self.scaling
        return out

__all__ = ["QuantumNATEnhanced"]
