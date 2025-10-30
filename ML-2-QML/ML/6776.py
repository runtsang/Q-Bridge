"""Hybrid estimator combining classical CNN features with a TorchQuantum variational layer.

This module defines a PyTorch model that first extracts features using a lightweight
convolutional network (inspired by Quantum‑NAT) and then maps those features into
a quantum device via a parameterized circuit. The output is a regression prediction
obtained from the expectation value of a Pauli observable, followed by a final linear
layer. The design allows easy substitution of the quantum backend and demonstrates
how classical and quantum sub‑networks can be jointly trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class ClassicalFeatureExtractor(nn.Module):
    """Lightweight CNN that reduces 28×28 grayscale images to a 64‑dim vector."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        feats = feats.view(bsz, -1)
        return self.fc(feats)


class QuantumLayer(tq.QuantumModule):
    """Variational layer that operates on 4 qubits."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.cnot = tq.CNOT(has_params=False, trainable=False)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=2)
        self.cnot(qdev, wires=[0, 3])
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)


class HybridEstimatorQNN(nn.Module):
    """
    Hybrid classical‑quantum estimator.

    The forward pass consists of:
      1. Classical CNN feature extraction → 64‑dim vector.
      2. Quantum encoding of the features into a 4‑qubit device.
      3. Variational quantum circuit (QuantumLayer).
      4. Measurement of Pauli‑Z on all qubits.
      5. Batch‑norm and a final linear layer to produce a scalar output.
    """
    def __init__(self) -> None:
        super().__init__()
        self.classical = ClassicalFeatureExtractor()
        self.quantum = QuantumLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.quantum.n_wires)
        self.out = nn.Linear(self.quantum.n_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Classical feature extraction
        feat = self.classical(x)          # (bsz, 64)
        # Encode features into qubits (first 4 qubits)
        qdev = tq.QuantumDevice(n_wires=self.quantum.n_wires,
                                bsz=bsz,
                                device=x.device,
                                record_op=True)
        # Map the first 4 feature values to rotation angles
        for i in range(self.quantum.n_wires):
            tqf.rx(qdev, params=feat[:, i], wires=i)
        # Apply variational layer
        self.quantum(qdev)
        # Measure expectation values
        out = self.measure(qdev)          # (bsz, 4)
        out = self.norm(out)
        return self.out(out).squeeze(-1)


__all__ = ["HybridEstimatorQNN"]
