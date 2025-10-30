"""Hybrid classical‑quantum regression – classical side.

The module mirrors the original QuantumRegression anchor but augments the
classical model with a quantum sub‑module that can be toggled on or off.
It demonstrates how a dense feature extractor can feed a variational
quantum circuit and then map the quantum measurement back into a scalar
prediction.

The design is intentionally modular: each component can be replaced
independently, facilitating ablation studies or further scaling.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchquantum as tq


# ------------------------------------------------------------------
# Dataset helpers – identical to the original seed, but with a
# deterministic seed for reproducibility.
# ------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data in the form of a superposition of |0…0> and |1…1>."""
    rng = np.random.default_rng(12345)
    angles = rng.uniform(-np.pi, np.pi, size=samples)
    phi = rng.uniform(-np.pi, np.pi, size=samples)

    # Build the state vector for each sample
    states = np.zeros((samples, 2 ** num_features), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(angles[i]) * np.eye(2 ** num_features)[0] \
                    + np.exp(1j * phi[i]) * np.sin(angles[i]) * np.eye(2 ** num_features)[-1]

    # Target is a smooth function of the angles
    labels = np.sin(2 * angles) * np.cos(phi)
    return states.astype(np.complex64), labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns a tensor of complex states and a scalar target."""

    def __init__(self, samples: int, num_features: int):
        self.states, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# ------------------------------------------------------------------
# Classical dense backbone – a shallow MLP.
# ------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    """A small MLP that turns the raw state amplitudes into a feature vector."""

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------
# Quantum sub‑module – a QCNN‑style variational circuit.
# ------------------------------------------------------------------
class QConvLayer(tq.QuantumModule):
    """Quantum equivalent of a single convolutional layer."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Parameterised rotations per wire
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        # Entangling gates
        self.cx = tq.CX()

    def forward(self, qdev: tq.QuantumDevice) -> None:
        # Apply rotations
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
        # Entangle neighbouring qubits
        for wire in range(self.n_wires - 1):
            self.cx(qdev, control=wire, target=wire + 1)


class QPoolLayer(tq.QuantumModule):
    """Quantum pooling that reduces wire count by discarding the last qubit."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Keep first half of wires, drop the rest using a simple measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        # Measure all wires – we keep only the first half of the results
        full_features = self.measure(qdev)
        return full_features[:, : self.n_wires // 2]


class QCNNQuantumModule(tq.QuantumModule):
    """QCNN‑style quantum circuit composed of conv and pool layers."""

    def __init__(self, num_wires: int, depth: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        current_wires = num_wires
        for _ in range(depth):
            self.layers.append(QConvLayer(current_wires))
            self.layers.append(QPoolLayer(current_wires))
            current_wires //= 2  # Pooling halves the number of wires

        # Final measurement to produce a feature vector
        self.final_measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, QConvLayer):
                layer(qdev)
            elif isinstance(layer, QPoolLayer):
                layer(qdev)
        return self.final_measure(qdev)


# ------------------------------------------------------------------
# Hybrid model – classical extractor + quantum QCNN + linear head.
# ------------------------------------------------------------------
class HybridRegressionModel(nn.Module):
    """End‑to‑end model that mixes classical and quantum transformations."""

    def __init__(self, num_features: int, num_qubits: int, use_quantum: bool = True):
        super().__init__()
        self.use_quantum = use_quantum
        self.extractor = FeatureExtractor(num_features, hidden_dim=64)

        if use_quantum:
            self.quantum = QCNNQuantumModule(num_qubits)
            # Linear head maps quantum measurements to a scalar
            self.head = nn.Linear(num_qubits // (2 ** 3), 1)  # matches final wire count
        else:
            self.head = nn.Linear(64, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # Classical feature extraction
        features = self.extractor(states)

        if self.use_quantum:
            # Map classical features into a quantum device
            batch_size = states.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.quantum.layers[0].n_wires,
                                    bsz=batch_size,
                                    device=states.device)
            # Use a simple linear encoder – here we reuse the classical features
            # as amplitudes (requires normalization)
            encoded = torch.nn.functional.normalize(features, dim=1)
            qdev.set_state(encoded)

            # Forward through QCNN
            q_features = self.quantum(qdev)
            # Collapse to scalar
            return self.head(q_features).squeeze(-1)
        else:
            return self.head(features).squeeze(-1)


# Expose the model under the original anchor name
QModel = HybridRegressionModel
__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
