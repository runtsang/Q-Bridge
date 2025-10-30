"""QuantumNATEnhanced: Quantum module with trainable encoder and variational layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum module that encodes classical data into a 4‑qubit state, applies a variational circuit, and reads out expectation values."""
    class QLayer(tq.QuantumModule):
        """Variational layer with trainable rotations and entanglement."""
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            # Random layer for initial entanglement
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
            # Parameterized single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Entangling gates
            self.cnot = tq.CNOT

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            # Entangle qubits in a ring
            for w in range(self.n_wires):
                self.cnot(qdev, wires=[w, (w+1)%self.n_wires])

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encoder: 4‑qubit parameterized rotation encoding
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Multi‑scale pooling of the input image
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
