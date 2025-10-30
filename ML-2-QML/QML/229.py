"""QuantumNATModel: Quantum variant with a two‑layer variational ansatz and learnable entangling gates."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATModel(tq.QuantumModule):
    """Quantum neural network with a two‑layer variational ansatz and learnable entangling gates."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            # Parameter‑efficient ansatz: single‑qubit rotations + controlled‑Z entanglement
            self.rx_params = nn.Parameter(torch.randn(n_wires))
            self.ry_params = nn.Parameter(torch.randn(n_wires))
            self.rz_params = nn.Parameter(torch.randn(n_wires))
            self.cZ_params = nn.Parameter(torch.randn(n_wires * (n_wires - 1) // 2))
            # Gate definitions
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.cZ = tq.CZ(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # Single‑qubit rotations
            for w in range(self.n_wires):
                self.rx(qdev, wires=w, params=self.rx_params[w])
                self.ry(qdev, wires=w, params=self.ry_params[w])
                self.rz(qdev, wires=w, params=self.rz_params[w])
            # Entangling layer
            idx = 0
            for i in range(self.n_wires):
                for j in range(i + 1, self.n_wires):
                    self.cZ(qdev, wires=[i, j], params=self.cZ_params[idx])
                    idx += 1

    def __init__(self, n_wires: int = 4, n_classes: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Encoder: simple linear mapping from pooled image features to qubit states
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=True)
        # Pool image features
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, -1)  # shape (batch, 16)
        # Encode into qubits
        self.encoder(qdev, pooled)
        # Variational circuit
        self.q_layer(qdev)
        # Measurement
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QuantumNATModel"]
