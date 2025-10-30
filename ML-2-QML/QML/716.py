import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class EnhancedQFCModel(tq.QuantumModule):
    """Variational quantum circuit with classical encoder and fully‑connected head."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 4, n_layers: int = 3):
            super().__init__()
            self.n_wires = n_wires
            self.n_layers = n_layers
            # Parameterised single‑qubit rotations
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rx = tq.RX(has_params=True, trainable=True)
            # Entangling gates
            self.crx = tq.CRX(has_params=True, trainable=True)
            # Random initial layer for feature scrambling
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for _ in range(self.n_layers):
                # Single‑qubit rotations
                for w in range(self.n_wires):
                    self.ry(qdev, wires=w)
                    self.rz(qdev, wires=w)
                # Cyclic entanglement
                for i in range(self.n_wires):
                    j = (i + 1) % self.n_wires
                    self.crx(qdev, wires=[i, j])
            # Optional Hadamard on the last qubit
            tqf.hadamard(qdev, wires=self.n_wires - 1,
                         static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_wires: int = 4, n_layers: int = 3, num_classes: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.num_classes = num_classes
        # Encoder mapping 16 classical features to 4 qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires=self.n_wires, n_layers=n_layers)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["EnhancedQFCModel"]
