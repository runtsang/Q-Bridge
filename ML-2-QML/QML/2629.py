import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class HybridFCModel(tq.QuantumModule):
    """
    Quantum version of the hybrid model.
    Architecture:
        CNN feature extractor -> average pooling -> quantum encoder -> variational quantum layer -> measurement -> batch norm.
    The quantum layer is a parameterized circuit with random gates and trainable rotations,
    closely following the QLayer in the QuantumNAT reference.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=3)
            self.crx(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Classical CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Quantum encoder for the pooled features
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Variational quantum layer
        self.q_layer = self.QLayer(n_wires=n_wires)
        # Measurement of all qubits in the Pauli-Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Batch normalization on the quantum expectation vector
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Prepare quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Classical feature pooling
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Encode classical features into quantum state
        self.encoder(qdev, pooled)
        # Apply variational layer
        self.q_layer(qdev)
        # Measure expectation values
        out = self.measure(qdev)
        # Normalize
        return self.norm(out)

def FCL() -> HybridFCModel:
    """Return an instance of the hybrid quantum model."""
    return HybridFCModel()

__all__ = ["HybridFCModel", "FCL"]
