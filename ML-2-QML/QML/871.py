"""Enhanced quantum model with parameter‑efficient ansatz and classical‑quantum fusion."""
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F

class QLayer(tq.QuantumModule):
    """Parameter‑efficient hardware‑efficient ansatz."""
    def __init__(self, n_wires: int = 4, depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        # Use a simple circuit: RY on each wire, followed by CNOT chain
        self.ry_params = nn.Parameter(torch.randn(depth, n_wires))
        self.cnot_pattern = [(i, (i+1)%n_wires) for i in range(n_wires)]

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        for d in range(self.depth):
            for w in range(self.n_wires):
                tqf.ry(qdev, wires=w, params=self.ry_params[d, w],
                       static=self.static_mode, parent_graph=self.graph)
            for (ctrl, target) in self.cnot_pattern:
                tqf.cnot(qdev, wires=[ctrl, target],
                         static=self.static_mode, parent_graph=self.graph)

class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum model with classical encoder, efficient ansatz, and measurement."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Encoder: use a simple 4x4 RyZXY encoding
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = QLayer(n_wires=n_wires, depth=3)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Classical feature extraction: average pool to 16 features
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        # Encode classical features into quantum state
        self.encoder(qdev, pooled)
        # Apply efficient ansatz
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
