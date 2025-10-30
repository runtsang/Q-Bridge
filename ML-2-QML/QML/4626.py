import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Quantum kernel applied to 2×2 patches."""
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

    @tq.static_support
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

class EstimatorQNN(tq.QuantumModule):
    """Hybrid quantum model combining quanvolution, encoder, QLayer and linear head."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilterQuantum()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = tq.RandomLayer(n_ops=50, wires=list(range(4)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.fc = nn.Sequential(nn.Linear(4 * 14 * 14, 64), nn.ReLU(), nn.Linear(64, 1))
        self.norm = nn.BatchNorm1d(1)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.qfilter(x)  # quantum‑encoded patches
        qdev = tq.QuantumDevice(4, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, features)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.fc(out)
        return self.norm(out)
