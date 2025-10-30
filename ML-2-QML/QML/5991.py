import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum quanvolution filter that processes 2x2 patches."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuanvolutionRegressor(tq.QuantumModule):
    """Quantum regression model that uses quanvolution filter followed by a variational regression head."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
            return None

    def __init__(self, num_wires: int = 4) -> None:
        super().__init__()
        self.num_wires = num_wires
        self.qfilter = QuanvolutionFilter(num_wires)
        self.q_layer = self.QLayer(num_wires * 14 * 14)  # number of patches * 4
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires * 14 * 14, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)  # shape (B, num_patches*4)
        bsz = features.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.q_layer.n_wires, bsz=bsz, device=device)
        # Encode features into rotations
        for i in range(self.q_layer.n_wires):
            tq.RX(angles=features[:, i], wires=i)(qdev)
            tq.RY(angles=features[:, i], wires=i)(qdev)
        # Apply variational layer
        self.q_layer(qdev)
        # Measure
        measurement = self.measure(qdev)
        out = self.head(measurement)
        return out.squeeze(-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionRegressor"]
