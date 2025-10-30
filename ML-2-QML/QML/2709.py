import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class SamplerQNN(tq.QuantumModule):
    """Quantum sampler network that combines a convolutional encoder
    (from Quantum‑NAT) with a variational quantum circuit and
    a probability‑output head.  The module returns a distribution
    over two outcomes, mirroring the classical SamplerQNN."""
    class QLayer(tq.QuantumModule):
        """Variational layer with a random circuit followed by
        trainable rotations and a small entangling block."""
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.cnot = tq.CNOT

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.cnot(qdev, wires=[0, 3])
            self.cnot(qdev, wires=[1, 2])

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Classical encoder identical to the one in the ML version
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        # Measure the first qubit to obtain a single‑bit outcome
        self.measure = tq.Measure(tq.PauliZ, wires=[0])
        self.norm = nn.BatchNorm1d(1)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over two outcomes."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        exp = self.measure(qdev)
        exp = self.norm(exp)
        probs = (exp + 1) / 2
        return torch.cat([probs, 1 - probs], dim=1)

__all__ = ["SamplerQNN"]
