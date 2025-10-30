import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# ----------------------------------------------------------------------
# Quantum fully‑connected layer (from reference 2)
# ----------------------------------------------------------------------
class QLayer(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

# ----------------------------------------------------------------------
# Quantum convolution‑style layer (inspired by reference 3)
# ----------------------------------------------------------------------
class QConvLayer(tq.QuantumModule):
    """
    A lightweight quantum layer that mimics a convolution filter by applying
    a RandomLayer followed by a few parameterised single‑qubit rotations
    and a controlled‑NOT. It is deliberately simple yet expressive.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.cnot = tq.CNOT()

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=2)
        self.cnot(qdev, wires=[0, 2])

# ----------------------------------------------------------------------
# Full hybrid quantum model
# ----------------------------------------------------------------------
class HybridQuantumNAT(tq.QuantumModule):
    """
    Quantum counterpart of HybridNAT.
    Combines a quantum encoder, a convolution‑style quantum layer,
    the QLayer from reference 2, and measurement.
    """
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encoder: use a simple 4‑qubit 4×4 Ry‑Z‑X‑Y pattern
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Quantum convolution layer
        self.qconv = QConvLayer(self.n_wires)
        # Quantum fully‑connected layer
        self.qfc = QLayer()
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Classical pooling to match the 4‑qubit encoder input
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Encode the pooled features into the quantum state
        self.encoder(qdev, pooled)
        # Apply the convolution‑style quantum layer
        self.qconv(qdev)
        # Apply the quantum fully‑connected layer
        self.qfc(qdev)
        # Measurement and post‑processing
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["HybridQuantumNAT"]
