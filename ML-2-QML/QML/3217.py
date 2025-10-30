"""Quantum regression model that processes image patches with a quantum filter and a variational circuit."""
import torch
import torch.nn as nn
import torchquantum as tq

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum-aware filter that produces n_wires features per image via global average pooling."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.conv = nn.Conv2d(1, n_wires, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        features = self.conv(x)  # (batch, n_wires, 14, 14)
        features = torch.mean(features, dim=(2, 3))  # (batch, n_wires)
        return features

class QuanvolutionHybridModel(tq.QuantumModule):
    """Hybrid quantum regression model that uses a quanvolution filter followed by a variational circuit."""
    def __init__(self, n_wires: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.qfilter = QuanvolutionFilter(n_wires)
        # Encoder maps each feature to a rotation on the corresponding qubit
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.q_layer = self._build_q_layer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 1)

    def _build_q_layer(self, num_wires: int) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self, num_wires: int):
                super().__init__()
                self.n_wires = num_wires
                self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
                self.rx = tq.RX(has_params=True, trainable=True)
                self.ry = tq.RY(has_params=True, trainable=True)

            def forward(self, qdev: tq.QuantumDevice):
                self.random_layer(qdev)
                for wire in range(self.n_wires):
                    self.rx(qdev, wires=wire)
                    self.ry(qdev, wires=wire)
        return QLayer(num_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        features = self.qfilter(x)  # (batch, n_wires)
        self.encoder(qdev, features)
        self.q_layer(qdev)
        measurement = self.measure(qdev)
        return self.head(measurement).squeeze(-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionHybridModel"]
