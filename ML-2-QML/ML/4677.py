"""Hybrid classical regression model combining QCNN‑style layers and a quantum fully‑connected block.

The model consists of:
* a classical feature extractor (MLP),
* a QCNN‑inspired sequence of linear layers,
* a quantum fully‑connected block implemented with torchquantum.
It is compatible with the dataset generator from the original seed but adds dropout and a projection layer
to feed the quantum block.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchquantum as tq

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate classical features that mimic the structure of the quantum superposition
    used in the seed implementation but with additional noise and a sinusoidal
    interaction term to increase non‑linearity.
    """
    # Base features uniformly sampled
    base = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    # Add sinusoidal interaction term
    angles = base.sum(axis=1, keepdims=True)
    noise = 0.1 * np.random.randn(samples, 1).astype(np.float32)
    labels = np.sin(2 * angles) + 0.1 * np.cos(angles) + noise
    return base, labels.squeeze()

class RegressionDataset(Dataset):
    """Dataset wrapping the superposition data generator."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "features": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumFullyConnected(nn.Module):
    """Quantum fully‑connected block that applies a random layer followed by
    single‑qubit RX/RY rotations and measures Pauli‑Z expectation values.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode classical data into the device
        encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{self.n_wires}xRy"])
        encoder(qdev, state_batch)
        # Apply random and rotation layers
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)
        # Measure expectation values of Pauli‑Z on each wire
        measure = tq.MeasureAll(tq.PauliZ)
        return measure(qdev)

class HybridRegressionModel(nn.Module):
    """Hybrid classical‑quantum regression model.

    * Classical extractor: MLP with dropout.
    * QCNN‑style linear layers.
    * Quantum fully‑connected block.
    * Final linear head.
    """
    def __init__(self, num_features: int, num_qbits: int = 4):
        super().__init__()
        # Classical feature extractor
        self.extractor = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Projection to quantum dimension
        self.proj = nn.Linear(16, num_qbits)
        # Quantum fully‑connected block
        self.q_fc = QuantumFullyConnected(num_qbits)
        # Final head
        self.head = nn.Linear(num_qbits, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.extractor(state_batch)
        q_input = torch.tanh(self.proj(x))
        q_out = self.q_fc(q_input)
        return self.head(q_out).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
