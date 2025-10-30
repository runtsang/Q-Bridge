import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class HybridNATModel(tq.QuantumModule):
    """Hybrid quantum model combining a random encoder, a QLayer, and a photonic‑style layer."""
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

    class PhotonicLayer(tq.QuantumModule):
        """A second layer mimicking the photonic fraud‑detection circuit using qubit gates."""
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            # Trainable gates that emulate the sequence of BS, R, S, D, K operations
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)
            self.rx1 = tq.RX(has_params=True, trainable=True)
            self.ry1 = tq.RY(has_params=True, trainable=True)
            self.rz1 = tq.RZ(has_params=True, trainable=True)
            self.crx1 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # First block – mimics a beam‑splitter and phase shifters
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=2)
            self.crx0(qdev, wires=[0, 3])
            # Second block – another set of rotations
            self.rx1(qdev, wires=1)
            self.ry1(qdev, wires=2)
            self.rz1(qdev, wires=3)
            self.crx1(qdev, wires=[1, 0])
            # Final corrective gates
            tqf.hadamard(qdev, wires=0, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.photonic_layer = self.PhotonicLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        self.photonic_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["HybridNATModel"]
