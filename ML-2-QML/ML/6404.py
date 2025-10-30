"""Hybrid classical‑quantum regressor using PyTorch and TorchQuantum.

The model consists of:
  * a classical encoder that transforms raw features into rotation angles,
  * a quantum layer (random unitary + trainable RX/RY) implemented with TorchQuantum,
  * a measurement of Pauli‑Z on all qubits,
  * a linear head mapping the expectation values to a scalar prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

# --------------------------------------------------------------------------- #
# Data utilities – synthetic regression dataset
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with a noisy sinusoidal target."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that yields a feature vector and a scalar target."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Hybrid model – classical encoder → quantum layer → classical head
# --------------------------------------------------------------------------- #
class HybridRegressor(nn.Module):
    """Hybrid classical‑quantum regression network."""
    def __init__(self, num_features: int, num_wires: int = 4):
        super().__init__()
        # Classical encoder: map inputs to rotation angles
        self.encoder = nn.Sequential(
            nn.Linear(num_features, num_wires * 2),
            nn.Tanh(),
        )
        # Quantum sub‑module
        self.q_layer = tq.QuantumModule(
            tq.RandomLayer(n_ops=30, wires=list(range(num_wires))),
            tq.RX(has_params=True, trainable=True),
            tq.RY(has_params=True, trainable=True),
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Encode classical input into rotation parameters
        params = self.encoder(state_batch)  # (bsz, 2*num_wires)
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.q_layer[0].n_wires,
            bsz=bsz,
            device=state_batch.device,
        )
        # Apply trainable rotations
        for i in range(self.q_layer[1].n_wires):
            self.q_layer[1](qdev, params[:, i], wires=i)
            self.q_layer[2](qdev, params[:, i + self.q_layer[1].n_wires], wires=i)
        # Random unitary
        self.q_layer[0](qdev)
        # Measurement
        features = self.measure(qdev)
        # Classical head
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressor", "RegressionDataset", "generate_superposition_data"]
