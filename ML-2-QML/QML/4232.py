import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Quantum patch‑wise filter inspired by the classical quanvolution."""
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


class QuantumHead(tq.QuantumModule):
    """Variational quantum head that maps flattened features to 4 qubit expectations."""
    def __init__(self, in_features: int = 784):
        super().__init__()
        self.n_wires = 4
        self.fc = nn.Linear(in_features, self.n_wires)
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        encoded = self.fc(x)
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        self.encoder(qdev, encoded)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


class QuantumNATGen223(tq.QuantumModule):
    """Hybrid quantum architecture mirroring the classical counterpart."""
    def __init__(self):
        super().__init__()
        # Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Quantum quanvolution filter
        self.qfilter = QuanvolutionFilterQuantum()
        # Quantum variational head
        self.head = QuantumHead(in_features=4 * 14 * 14)  # 784

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.qfilter(x)
        x = self.head(x)
        return x


__all__ = ["QuanvolutionFilterQuantum", "QuantumHead", "QuantumNATGen223"]
