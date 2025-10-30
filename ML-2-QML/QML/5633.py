import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNAT__gen004(tq.QuantumModule):
    """
    Extended quantum model based on the original QFCModel.
    Adds a variable‑depth random layer, a tunable number of
    variational gates, and an optional classical post‑processing
    head.  The model remains compatible with the original
    interface.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, depth: int, n_ops: int):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            # Random layer with tunable number of ops
            self.random_layer = tq.RandomLayer(
                n_ops=n_ops, wires=list(range(self.n_wires)), depth=self.depth
            )
            # Variational rotation gates (one per wire)
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # Apply the random layer multiple times
            for _ in range(self.depth):
                self.random_layer(qdev)
            # Apply variational rotations
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            # Basic entanglement
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)
            # Optional Hadamard and SX gates
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)

    def __init__(self,
                 n_wires: int = 4,
                 encoder_name: str = "4x4_ryzxy",
                 quantum_depth: int = 1,
                 random_n_ops: int = 50,
                 use_classical_head: bool = True):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[encoder_name])
        self.q_layer = self.QLayer(n_wires=self.n_wires,
                                   depth=quantum_depth,
                                   n_ops=random_n_ops)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.use_classical_head = use_classical_head
        if self.use_classical_head:
            # Classical post‑processing head
            self.classical_post = nn.Sequential(
                nn.Linear(self.n_wires, 8),
                nn.ReLU(inplace=True),
                nn.Linear(8, 4),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                bsz=bsz,
                                device=x.device,
                                record_op=True)
        # Pool the input to match the encoder dimensionality
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out)
        if self.use_classical_head:
            out = self.classical_post(out)
        return out

__all__ = ["QuantumNAT__gen004"]
