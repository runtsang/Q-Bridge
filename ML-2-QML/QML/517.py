import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionFilter(tq.QuantumModule):
    """Variational quanvolution: data‑encoded rotations followed by a learnable circuit."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Data encoding with Ry rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Learnable variational layer (8 parameters)
        self.param_layer = tq.ParametricLayer(
            [
                {"func": "ry", "wires": [0]},
                {"func": "ry", "wires": [1]},
                {"func": "ry", "wires": [2]},
                {"func": "ry", "wires": [3]},
                {"func": "cx", "control_wires": [0], "target_wires": [1]},
                {"func": "cx", "control_wires": [2], "target_wires": [3]},
            ],
            n_params=8,
        )
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
                self.param_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid network: variational quanvolution + classical linear head."""
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, n_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.qfilter(x)
        logits = self.linear(feats)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
