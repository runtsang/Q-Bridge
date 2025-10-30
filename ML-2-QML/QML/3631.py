import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class HybridNATModel(tq.QuantumModule):
    """Quantum‑aware model inspired by Quantum‑NAT and EstimatorQNN.

    The circuit encodes pooled image features with a GeneralEncoder,
    applies a variational QLayer, and then measures all qubits.  The
    measurement result is passed through a classical dense head to
    produce a scalar regression output.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=30, wires=list(range(self.n_wires))
            )
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head to produce regression output
        self.head = nn.Sequential(
            nn.Linear(self.n_wires, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.head(out)
        return self.norm(out)

__all__ = ["HybridNATModel"]
