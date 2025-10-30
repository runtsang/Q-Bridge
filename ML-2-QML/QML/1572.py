import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNatHybrid(tq.QuantumModule):
    """Quantum variant of the hybrid architecture."""
    class QLayer(tq.QuantumModule):
        def __init__(self, depth: int, n_wires: int):
            super().__init__()
            self.depth = depth
            self.n_wires = n_wires
            self.layers = nn.ModuleList()
            for _ in range(depth):
                layer = tq.QuantumModule()
                layer.rand = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
                layer.rx = tq.RX(has_params=True, trainable=True)
                layer.ry = tq.RY(has_params=True, trainable=True)
                layer.rz = tq.RZ(has_params=True, trainable=True)
                layer.crx = tq.CRX(has_params=True, trainable=True)
                self.layers.append(layer)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            for layer in self.layers:
                layer.rand(qdev)
                layer.rx(qdev, wires=0)
                layer.ry(qdev, wires=1)
                layer.rz(qdev, wires=2)
                layer.crx(qdev, wires=[0, 2])
                tqf.hadamard(qdev, wires=0, static=self.static_mode, parent_graph=self.graph)
                tqf.cnot(qdev, wires=[1, 3], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_qubits: int = 4, depth: int = 3, use_residual: bool = True):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.use_residual = use_residual
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(depth, n_qubits)
        self.measure = tq.MeasureAll(tq.PauliZ)
        if use_residual:
            self.skip_proj = nn.Linear(16, n_qubits)
        else:
            self.skip_proj = None
        self.norm = nn.BatchNorm1d(n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)  # 4×4 grid → 16 features
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        if self.use_residual:
            skip = self.skip_proj(pooled)
            out = out + skip
        return self.norm(out)

__all__ = ["QuantumNatHybrid"]
