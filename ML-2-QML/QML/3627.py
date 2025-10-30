"""Hybrid quantum‑classical regression model.

The quantum part is a variational circuit that takes the features extracted
by a CNN (inherited from the classical version) and produces a single‑qubit
output that is mapped to a real‑valued regression target.  The architecture
inherits the random layer, RX/RY/RZ/CRX gates of the Quantum‑NAT example
and the GeneralEncoder used in the original quantum regression script.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data; identical to the classical version."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

def _features_to_image(features: torch.Tensor, pixel_dim: int | None = None) -> torch.Tensor:
    """Same helper as in the classical module."""
    batch, num_features = features.shape
    if pixel_dim is None:
        pixel_dim = int(np.ceil(np.sqrt(num_features)))
    padded = F.pad(features, (0, pixel_dim * pixel_dim - num_features))
    return padded.view(batch, 1, pixel_dim, pixel_dim)

class RegressionDataset(Dataset):
    """Same dataset as in the classical module."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridRegressionModel(tq.QuantumModule):
    """Quantum‑enhanced regression model with a CNN feature extractor."""

    class QLayer(tq.QuantumModule):
        """Variational circuit that adds entanglement and single‑qubit rotations."""

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(
                n_ops=30, wires=list(range(n_wires)), trainable=False
            )
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            # entanglement pattern similar to the QLayer in Quantum‑NAT
            self.crx(qdev, wires=[0, 1])
            self.crx(qdev, wires=[2, 3])
            tqf.hadamard(qdev, wires=0, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=1, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[0, 2], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, num_features: int):
        super().__init__()
        self.pixel_dim = int(np.ceil(np.sqrt(num_features)))
        self.n_wires = 4  # fixed number of qubits for the feature encoding
        # Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Quantum block
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{self.n_wires}xRy"])
        self.q_layer = self.QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head
        self.head = nn.Linear(self.n_wires, 1)
        self.norm = nn.BatchNorm1d(1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        img = _features_to_image(state_batch, self.pixel_dim)
        bsz = img.shape[0]
        feats = self.features(img)
        pooled = F.avg_pool2d(feats, 6).view(bsz, 16)  # matching Quantum‑NAT pooling
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device, record_op=True)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        features_q = self.measure(qdev)
        out = self.head(features_q)
        return self.norm(out).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
