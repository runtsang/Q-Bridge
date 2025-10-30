import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATGen065(tq.QuantumModule):
    """Quantum regression model that mirrors the classical counterpart.
    Uses a 4‑wire encoder, a random variational layer, and a linear regression head.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 1)
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Reduce image to a flat vector per sample
        pooled = F.avg_pool2d(x, 6).view(bsz, -1)
        # Ensure input matches number of wires
        if pooled.shape[1]!= self.n_wires:
            pooled = torch.nn.functional.pad(pooled, (0, self.n_wires - pooled.shape[1]), "constant", 0)
            pooled = pooled[:, :self.n_wires]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        features = self.measure(qdev)
        out = self.head(features)
        return self.norm(out).squeeze(-1)

__all__ = ["QuantumNATGen065"]
