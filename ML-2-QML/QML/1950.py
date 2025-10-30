"""Quantum-enhanced quanvolution with trainable linear mapping and parametric random layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum quanvolution filter with trainable linear mapping and parametric random layer."""
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
        # Trainable linear mapping from 4 pixel values to 4 rotation angles
        self.parametric_encoder = nn.Linear(4, 4)
        # Parametric random layer (trainable)
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
                # classical linear mapping to angles
                angles = self.parametric_encoder(data)
                self.encoder(qdev, angles)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid quantum-classical model with transformer encoder."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        encoder_layer = TransformerEncoderLayer(
            d_model=4 * 14 * 14, nhead=4, dim_feedforward=512, dropout=0.1
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        seq = features.unsqueeze(0)
        transformed = self.transformer(seq).squeeze(0)
        logits = self.linear(transformed)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
