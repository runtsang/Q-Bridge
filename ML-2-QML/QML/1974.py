import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import List

class QuantumNATExtended(tq.QuantumModule):
    """
    Quantum version of the extended Quantum‑Nat model. The circuit depth
    is configurable and each layer consists of a random layer followed by
    trainable single‑qubit rotations and fixed two‑qubit gates.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, depth: int = 2):
            super().__init__()
            self.blocks = nn.ModuleList()
            for _ in range(depth):
                block = nn.ModuleList([
                    tq.RandomLayer(n_ops=20, wires=[0, 1, 2, 3]),
                    tq.RX(has_params=True, trainable=True),
                    tq.RY(has_params=True, trainable=True),
                    tq.RZ(has_params=True, trainable=True),
                    tq.CRX(has_params=True, trainable=True, wires=[0, 2]),
                    tq.CNOT(wires=[1, 3]),
                ])
                self.blocks.append(block)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            for block in self.blocks:
                for op in block:
                    op(qdev)

    def __init__(self, n_wires: int = 4, depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(depth=depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.bn = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True
        )
        # Encode the image as a 4‑dimensional feature vector
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.bn(out)

    def fit(self, data_loader, lr: float = 1e-3, epochs: int = 10, device: str = 'cpu'):
        """Simple training loop for the quantum module."""
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.train()
        for epoch in range(epochs):
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = self(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
