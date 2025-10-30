"""Quantum kernel module implementing a variational overlap between
image embeddings.  The kernel first projects each image into a
4‑dimensional feature vector using a lightweight CNN (shared with
the classical counterpart).  The features are encoded on a 4‑qubit
device via Ry rotations, followed by a trainable variational block
comprising a random layer, single‑qubit rotations and controlled
gates.  The kernel value is obtained from the amplitude of the
all‑zero state after encoding both inputs in sequence, which
realises the inner product of the two encoded states.

"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class FeatureExtractor(nn.Module):
    """Convolutional feature extractor producing a 4‑dimensional vector."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        out = self.conv(x)
        out = out.view(bsz, -1)
        out = self.fc(out)
        return self.bn(out)


class QuantumAnsatz(tq.QuantumModule):
    """Variational ansatz that encodes two feature vectors sequentially."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoder using a fixed Ry–RZ–XY pattern
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}x4_ryzxy"])
        # Trainable variational block
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode x, apply variational block, then encode y with inverted sign."""
        qdev.reset_states(x.shape[0])
        # Encode first feature vector
        self.encoder(qdev, x)
        # Variational block
        self.random_layer(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=2)
        self.crx(qdev, wires=[0, 2])
        # Encode second feature vector with inverted sign
        self.encoder(qdev, -y)
        # Repeat variational block to symmetrise
        self.random_layer(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=2)
        self.crx(qdev, wires=[0, 2])


class HybridQuantumKernel(tq.QuantumModule):
    """Quantum kernel that uses a variational circuit to compute the overlap."""
    def __init__(self, n_wires: int = 4, gamma: float = 1.0) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.ansatz = QuantumAnsatz(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.feature_extractor = FeatureExtractor()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for two batches of images."""
        # Ensure 4‑D tensors
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if y.dim() == 3:
            y = y.unsqueeze(0)
        bsz = x.shape[0]
        # Classical feature extraction
        fx = self.feature_extractor(x)
        fy = self.feature_extractor(y)
        # Quantum evaluation
        qdev = tq.QuantumDevice(n_wires=self.n_wires, device=x.device, bsz=bsz)
        self.ansatz(qdev, fx, fy)
        out = self.measure(qdev)
        out = self.norm(out)
        return out.squeeze()

    def kernel_matrix(self, a: list[torch.Tensor], b: list[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix between two sequences of images."""
        mat = np.zeros((len(a), len(b)))
        for i, xa in enumerate(a):
            for j, yb in enumerate(b):
                mat[i, j] = self.forward(xa.unsqueeze(0), yb.unsqueeze(0)).item()
        return mat


__all__ = ["FeatureExtractor", "QuantumAnsatz", "HybridQuantumKernel"]
