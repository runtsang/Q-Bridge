import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionNet(tq.QuantumModule):
    """Hybrid quantum‑classical quanvolution network."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.n_wires = 4
        # Encode each 2×2 patch into a 4‑qubit circuit
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Trainable random layer
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Linear head
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        patches = []
        x = x.view(bsz, 28, 28)
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
        features = torch.cat(patches, dim=1)  # (B, 4*14*14)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
