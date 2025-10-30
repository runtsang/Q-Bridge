import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """
    Quantum branch of the enhanced Quantum‑NAT model.
    Uses a deeper variational circuit with a richer encoding and multi‑wire measurement.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            # Random layer with more operations
            self.random_layer = tq.RandomLayer(n_ops=80, wires=list(range(self.n_wires)))
            # Parameterized single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Two‑qubit entangling block
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # Apply random layer
            self.random_layer(qdev)
            # Apply single‑qubit rotations on each wire
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            # Entangle wires in a ring
            for w in range(self.n_wires):
                self.crx(qdev, wires=[w, (w + 1) % self.n_wires])
                tqf.cnot(qdev, wires=[w, (w + 1) % self.n_wires])
            # Add a few Hadamard gates for mixing
            tqf.hadamard(qdev, wires=range(self.n_wires))

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Use a richer encoder: 4x4_RYZX
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzx"])
        self.q_layer = self.QLayer()
        # Measure all wires in PauliZ basis
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Preprocess: average pool to match 16 features
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Encode the pooled features into qubits
        self.encoder(qdev, pooled)
        # Variational layer
        self.q_layer(qdev)
        # Measurement
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
