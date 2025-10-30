"""Advanced quantum model with a deeper, parameter‑shared entangling layer and measurement‑based loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridQFCModel(tq.QuantumModule):
    """Quantum encoder featuring a deeper circuit and shared entangling parameters."""
    class EntanglingLayer(tq.QuantumModule):
        """Parameter‑shared entangling circuit across all wires."""
        def __init__(self, n_wires: int, n_layers: int = 3):
            super().__init__()
            self.n_wires = n_wires
            self.n_layers = n_layers
            # One rotation angle per wire per layer for RX, RY, RZ
            self.params = nn.Parameter(torch.randn(n_layers, n_wires, 3))

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            for layer in range(self.n_layers):
                for wire in range(self.n_wires):
                    tqf.rx(qdev, self.params[layer, wire, 0], wires=wire)
                    tqf.ry(qdev, self.params[layer, wire, 1], wires=wire)
                    tqf.rz(qdev, self.params[layer, wire, 2], wires=wire)
                # Full‑ladder entanglement
                for i in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[i, i + 1])

    def __init__(self, n_wires: int = 4, num_classes: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.random_layer = tq.RandomLayer(n_ops=100, wires=list(range(n_wires)))
        self.entangling = self.EntanglingLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        # Classical feature pooling
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, -1)
        self.encoder(qdev, pooled)
        self.random_layer(qdev)
        self.entangling(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["HybridQFCModel"]
