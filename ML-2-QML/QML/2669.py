import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATHybridQML(tq.QuantumModule):
    """
    Quantum variant of the hybrid architecture: CNN‑encoded features are processed by a variational quantum circuit
    and a quantum autoencoder.
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

    class AutoEncoderLayer(tq.QuantumModule):
        """
        Simple quantum auto‑encoder that maps n_qubits → n_qubits via a variational circuit.
        """
        def __init__(self, n_qubits: int = 4):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.RandomLayer(n_ops=20, wires=list(range(n_qubits)))
            self.decoder = tq.RandomLayer(n_ops=20, wires=list(range(n_qubits)))

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.encoder(qdev)
            self.decoder(qdev)

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.feature_layer = self.QLayer()
        self.auto_layer = self.AutoEncoderLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.feature_layer(qdev)
        self.auto_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QuantumNATHybridQML"]
