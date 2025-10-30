import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class HybridFCL(tq.QuantumModule):
    """Quantum version of the Hybrid Fully Connected Layer.

    The module receives classical image tensors, performs a coarse average
    pooling to match the number of qubits, encodes the pooled values into a
    quantum device using a 4‑qubit Ry–Rz‑xy encoder, applies a short
    variational circuit (random layer + parameterised gates), measures all
    qubits in the Pauli‑Z basis, and normalises the resulting expectation
    values.  The overall structure mirrors the classical branch while
    exploiting quantum parallelism for the final projection.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))
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

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Encoder that maps 4‑dim pooled features to quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        # Classical pooling to produce a 4‑dim vector per sample
        pooled = F.avg_pool2d(x, 6).view(bsz, self.n_wires)
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True
        )
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["HybridFCL"]
