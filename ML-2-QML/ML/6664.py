"""Hybrid classical‑quantum feature extractor inspired by Quantum‑NAT.

This module defines a `QFCModelHybrid` class that can be used with either
PyTorch or a TorchQuantum‑compatible simulator.  The classical part is a
ResNet‑style bottleneck that preserves spatial resolution, while the
quantum part is a parameter‑reversible circuit that runs on a
`QuantumDevice` from TorchQuantum.  The class exposes a `forward` method
that accepts a batch of 1‑channel images and returns a 4‑dimensional
feature vector ready for a classifier or downstream task.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QFCModelHybrid(nn.Module):
    """Hybrid classical‑quantum architecture for image‑based tasks."""

    def __init__(self, n_wires: int = 4, n_layers: int = 3):
        super().__init__()
        # Classical feature extractor: 3‑layer ResNet‑bottleneck
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        # Linear bottleneck to match quantum dimension
        self.bottleneck = nn.Linear(16 * 28 * 28, 4 * n_wires)
        self.n_wires = n_wires
        self.n_layers = n_layers
        # Learnable parameters for each variational layer
        self.params = nn.Parameter(torch.randn(n_layers, n_wires, 3))
        self.norm = nn.BatchNorm1d(n_wires)
        # Quantum measurement module
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Classical feature extraction
        feats = self.features(x)                # (bsz, 16, 28, 28)
        flat = feats.view(bsz, -1)              # (bsz, 16*28*28)
        # Linear bottleneck to prepare qubit amplitudes
        prep = self.bottleneck(flat)             # (bsz, 4*n_wires)
        # Reshape to (bsz, n_wires, 4) – each wire gets 4 real params
        angles = prep.view(bsz, self.n_wires, 4)
        # Quantum simulation using TorchQuantum
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True
        )
        # Initialise all qubits to |0>
        tqf.hadamard(qdev, wires=list(range(self.n_wires)), static=True, parent_graph=None)
        # Apply parameterised layers
        for i in range(self.n_layers):
            for w in range(self.n_wires):
                tqf.rx(qdev, self.params[i, w, 0], wires=w, static=True, parent_graph=None)
                tqf.ry(qdev, self.params[i, w, 1], wires=w, static=True, parent_graph=None)
                tqf.rz(qdev, self.params[i, w, 2], wires=w, static=True, parent_graph=None)
            # Entangle adjacent qubits
            for w in range(self.n_wires - 1):
                tqf.cx(qdev, wires=[w, w + 1], static=True, parent_graph=None)
        # Measure all qubits in Z basis
        out = self.measure(qdev)
        return self.norm(out)
