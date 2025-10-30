import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import func_name_dict
import torch.nn.functional as F


class QuantumFilter(tq.QuantumModule):
    """Quantum analogue of the classical 2‑D filter."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, data: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, data)
        self.random_layer(qdev)
        return self.measure(qdev)


class QuantumKernel(tq.QuantumModule):
    """Fixed quantum kernel based on a simple Ry encoding."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.qdev = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.qdev, x, y)
        return torch.abs(self.qdev.states.view(-1)[0])


class QFCLayer(tq.QuantumModule):
    """Learnable quantum fully‑connected block."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        return self.norm(self.measure(qdev))


class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum hybrid model: quantum filter → quantum kernel → quantum fully‑connected head.
    Designed to mirror the classical counterpart for fair benchmarking.
    """
    def __init__(self):
        super().__init__()
        self.filter = QuantumFilter()
        self.kernel = QuantumKernel()
        self.fc = QFCLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.filter.n_wires, bsz=bsz, device=x.device)

        # Extract 2x2 patches and apply quantum filter
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1
                )
                patches.append(self.filter(qdev, patch))
        feats = torch.cat(patches, dim=1)  # (batch, 784)

        # Quantum kernel on the feature vector (self‑similarity)
        diag = torch.diag(self.kernel(feats, feats)).unsqueeze(1)  # (batch, 1)
        combined = torch.cat([feats, diag], dim=1)  # (batch, 785)

        # Pass through the learnable quantum block
        out = self.fc(qdev)  # measurement returns (batch, 4)
        return F.log_softmax(out, dim=-1)


__all__ = ["QuantumFilter", "QuantumKernel", "QFCLayer", "QuanvolutionHybrid"]
