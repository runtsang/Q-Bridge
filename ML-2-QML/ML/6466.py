"""Advanced classical‑quantum hybrid model for MNIST classification.

This module extends the original Quanvolution example by adding:
* A trainable variational quantum circuit that is trainable via PyTorch autograd.
* A classical 1×1 convolution to post‑process the quantum‑derived features.
* New ``AdvancedQuanvolutionFilter`` and ``AdvancedQuanvolutionClassifier`` classes.

The original classes remain unchanged for backward compatibility.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchquantum import QuantumModule, QuantumDevice, MeasureAll, PauliZ, RandomLayer, GeneralEncoder


class QuanvolutionFilter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


class AdvancedQuanvolutionFilter(nn.Module):
    """Classical‑quantum hybrid filter: a trainable variational quantum circuit
    followed by a 1×1 classical convolution."""
    def __init__(self, n_qubits: int = 4, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.n_qubits = n_qubits
        # Quantum encoder
        self.encoder = GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Trainable variational layer
        self.var_layer = RandomLayer(n_ops=8, wires=list(range(n_qubits)), trainable=True)
        # Classical post‑processing
        self.post_conv = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = QuantumDevice(self.n_qubits, bsz=bsz, device=device)
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
                measurement = MeasureAll(PauliZ)(qdev)
                patches.append(measurement.view(bsz, 4))
        quantum_features = torch.cat(patches, dim=1)
        quantum_features = quantum_features.unsqueeze(-1)
        refined = self.post_conv(quantum_features).squeeze(-1)
        return refined


class AdvancedQuanvolutionClassifier(nn.Module):
    """Classifier that uses the AdvancedQuanvolutionFilter and a linear head."""
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
