"""Advanced quantum‑classical hybrid filter for MNIST classification.

This module extends the original Quanvolution example by adding a trainable
variational quantum circuit and a classical 1×1 convolution for post‑processing.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
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


class QuanvolutionClassifier(nn.Module):
    """Hybrid neural network using the quanvolutional filter followed by a linear head."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


class AdvancedQuanvolutionFilter(tq.QuantumModule):
    """Trainable variational quantum filter with classical post‑processing."""
    def __init__(self, n_wires: int = 4, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Trainable variational layer
        self.var_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)), trainable=True)
        # Classical post‑processing
        self.post_conv = nn.Conv1d(4, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        patches = []
        for r in range(0, 28, self.patch_size):
            for c in range(0, 28, self.patch_size):
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
                self.var_layer(qdev)
                measurement = tq.MeasureAll(tq.PauliZ)(qdev)
                patches.append(measurement.view(bsz, 4))
        quantum_features = torch.cat(patches, dim=1)
        quantum_features = quantum_features.unsqueeze(-1)
        refined = self.post_conv(quantum_features).squeeze(-1)
        return refined


class AdvancedQuanvolutionClassifier(nn.Module):
    """Classifier using the AdvancedQuanvolutionFilter."""
    def __init__(self):
        super().__init__()
        self.qfilter = AdvancedQuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = [
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "AdvancedQuanvolutionFilter",
    "AdvancedQuanvolutionClassifier",
]
