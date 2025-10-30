import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum quanvolution filter using a random two‑qubit kernel."""
    def __init__(self, n_wires: int = 4, n_ops: int = 8) -> None:
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
        self.q_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(self.n_wires)))
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

class SamplerQNN(tq.QuantumModule):
    """Quantum sampler network inspired by Qiskit SamplerQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 2
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
            ]
        )
        self.layer = tq.RandomLayer(n_ops=4, wires=[0, 1])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        self.encoder(qdev, x)
        self.layer(qdev)
        out = self.measure(qdev)
        return out

class QuanvolutionHybrid(tq.QuantumModule):
    """Quantum hybrid model that fuses a quanvolution filter with a sampler head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.sampler = SamplerQNN()
        self.classifier = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        # Use sampler on first two feature channels to produce gating scalar
        sampler_out = self.sampler(features[:, :2])
        gamma = sampler_out[:, 0].unsqueeze(-1)
        logits = self.classifier(features)
        gated_logits = logits * gamma
        return F.log_softmax(gated_logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
