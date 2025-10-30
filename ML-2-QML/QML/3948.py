import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuanvolutionHybridFilter(tq.QuantumModule):
    """Quantum version of the quanvolution filter that applies a random two‑qubit kernel to 2×2 patches."""
    def __init__(self) -> None:
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
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)   # (bsz, 4*14*14)

class QuanvolutionHybridClassifier(tq.QuantumModule):
    """Hybrid quantum‑classical classifier that uses a parametrised 4‑qubit circuit to map image patches to logits."""
    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])

    def __init__(self) -> None:
        super().__init__()
        self.filter = QuanvolutionHybridFilter()
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(4)
        self.linear = nn.Linear(4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Quantum feature extraction
        features = self.filter(x)  # (bsz, 4*14*14)
        aggregated = features.view(x.size(0), 14*14, 4).mean(dim=1)  # (bsz, 4)

        # Encode aggregated features into a 4‑qubit device
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.q_layer.n_wires, bsz=bsz, device=device)
        self.filter.encoder(qdev, aggregated)
        self.q_layer(qdev)
        quantum_out = self.measure(qdev)
        quantum_out = self.norm(quantum_out)

        # Classical linear head
        logits = self.linear(quantum_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybridFilter", "QuanvolutionHybridClassifier"]
