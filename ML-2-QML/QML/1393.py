import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum‑only branch of the enhanced hybrid model."""

    class VariationalLayer(tq.QuantumModule):
        """Simple parameterized rotation ansatz suitable for 4‑qubit circuits."""
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            # Each qubit gets a parameterized RX, RY, RZ
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Entangling layer
            self.cnot = tq.CNOT

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # Apply local rotations
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            # Entangle adjacent qubits
            for w in range(self.n_wires - 1):
                self.cnot(qdev, wires=[w, w + 1])
            # Final layer of rotations
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encoder that maps a 4‑dim classical embedding to qubit states
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.var_layer = self.VariationalLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Average‑pool to 4‑dim embedding (as in the seed)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.var_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
