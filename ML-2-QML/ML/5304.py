"""QuanvolutionHybrid: classical backbone with optional quantum feature extractor.

The module implements a flexible filter that can be either a pure PyTorch Conv2d
or a quantum kernel layer.  The quantum part is built with TorchQuantum
(``tq.QuantumModule``) and is fully differentiable.  The wrapper exposes a
``use_quantum`` flag that lets the user swap between the two regimes without
changing downstream code.  This design is inspired by the four reference
pairs:  * QuanvolutionFilter (classical 2‑D conv),  * Conv (drop‑in
replacement),  * QuantumKernelMethod (quantum kernel),  * QuantumRegression
(quantum regression head).  The combined class can be used in any pipeline
that expects a :class:`~torch.nn.Module`.

Typical usage::

    from quanvolution_gen190 import QuanvolutionHybrid
    model = QuanvolutionHybrid(use_quantum=True, num_qwires=4)
    # train on MNIST
    logits = model(x)          # shape (B, 10)

"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

__all__ = ["QuanvolutionHybrid", "QuanvolutionClassicalFilter", "QuanvolutionQuantumFilter", "QuanvolutionQuantumHead"]

class QuanvolutionClassicalFilter(nn.Module):
    """Classic 2×2 convolution filter that emulates the original quanvolution."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,28,28) -> (B,4,14,14) -> (B,4*14*14)
        return self.conv(x).view(x.size(0), -1)

class QuanvolutionQuantumFilter(tq.QuantumModule):
    """Quantum kernel based filter that processes 2×2 patches with a random circuit."""
    def __init__(self, num_wires: int = 4):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps each pixel value to a rotation about Y
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
        # x shape: (B,1,28,28)
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

class QuanvolutionQuantumHead(tq.QuantumModule):
    """Quantum head that maps aggregated patch features to logits."""
    def __init__(self, num_wires: int = 4):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=12, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(self.n_wires, 10)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, 4*14*14)
        B = features.shape[0]
        device = features.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=B, device=device)

        # aggregate 4 features per sample by averaging over all patches
        agg = features.view(B, -1, 4).mean(dim=1)
        self.encoder(qdev, agg)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.linear(out)

class QuanvolutionHybrid(nn.Module):
    """Hybrid model that can be run classically or with quantum layers."""
    def __init__(self, use_quantum: bool = True, use_quantum_head: bool = False, num_qwires: int = 4):
        super().__init__()
        self.use_quantum = use_quantum
        self.use_quantum_head = use_quantum_head

        self.classical_filter = QuanvolutionClassicalFilter()
        self.quantum_filter = QuanvolutionQuantumFilter(num_qwires)

        if use_quantum_head:
            self.head = QuanvolutionQuantumHead(num_qwires)
        else:
            self.head = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            features = self.quantum_filter(x)
        else:
            features = self.classical_filter(x)
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)
