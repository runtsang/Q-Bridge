"""Hybrid classical‑quantum regression model.

The architecture fuses a 2‑D CNN feature extractor, a 4‑qubit GeneralEncoder, a RandomLayer
with RX/RY/RZ/CRX gates, a measurement of all qubits, and a linear head for regression.
A `sample()` method demonstrates how the same quantum circuit can be used to produce
probabilities, akin to a SamplerQNN."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import Dataset
import numpy as np

# --------------------------------------------------------------------------- #
# Data utilities (borrowed from reference 3)
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class RegressionDataset(Dataset):
    """Dataset yielding complex quantum states and a regression target."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Hybrid model
# --------------------------------------------------------------------------- #
class HybridQuantumRegressionModel(nn.Module):
    """Hybrid CNN + Quantum layer + regression head.

    The model first extracts 4‑channel features from a 1‑channel image, encodes them
    into a 4‑qubit state via a GeneralEncoder, processes the state with a
    RandomLayer + RX/RY/RZ/CRX gates, measures all qubits and feeds the expectation
    values into a linear head.  A `sample()` method is provided to demonstrate
    how the same architecture can be used to produce probability distributions.
    """

    class _QuantumLayer(tq.QuantumModule):
        """Internal quantum layer used by the hybrid model."""
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            # add a few more gates for richer entanglement
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_wires: int = 4):
        super().__init__()
        # Classical CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Encoder that maps 16‑dim feature vector to 4 qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Quantum layer
        self.q_layer = self._QuantumLayer(n_wires)
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Linear head for regression
        self.head = nn.Linear(n_wires, 1)
        # Batch norm on final output for stability
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass: image -> CNN -> encoder -> quantum layer -> measurement -> head."""
        bsz = x.shape[0]
        # CNN feature extraction
        feats = self.features(x)  # shape (bsz, 16, 7, 7)
        # Global average pooling to get 16‑dim vector
        pooled = F.avg_pool2d(feats, kernel_size=feats.shape[2:]).view(bsz, -1)
        # Quantum device
        qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        features = self.measure(qdev)
        out = self.head(features)
        return self.norm(out).squeeze(-1)

    def sample(self, x: torch.Tensor, n_samples: int = 1000) -> torch.Tensor:
        """Return a probability distribution over 4 classes using a SamplerQNN style circuit."""
        bsz = x.shape[0]
        # Use a small quantum circuit to sample probabilities
        qdev = tq.QuantumDevice(n_wires=2, bsz=bsz, device=x.device)
        # Encode the pooled features into a 2‑qubit state
        pooled = F.avg_pool2d(self.features(x), kernel_size=x.shape[2:]).view(bsz, -1)
        # Simple encoding: rotate each qubit by the first two pooled values
        tqf.ry(qdev, angles=pooled[:, 0], wires=0)
        tqf.ry(qdev, angles=pooled[:, 1], wires=1)
        # Entangle
        tqf.cx(qdev, wires=[0, 1])
        # Measure in computational basis
        probs = tq.MeasureAll(tq.PauliZ)(qdev)
        # Convert expectation values to probabilities
        probs = (probs + 1) / 2
        return probs

__all__ = ["HybridQuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
