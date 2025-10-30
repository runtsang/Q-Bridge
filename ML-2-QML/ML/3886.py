"""QuantumHybridRegressor – classical component of the hybrid regression pipeline.

The code merges the two seed regressors into a single, trainable model:
* A `RegressionDataset` that can output either real‑valued features or pre‑encoded complex states.
* A `ClassicalEncoder` that maps real features to a complex amplitude vector.
* A `QuantumLayer` that implements a random layer followed by trainable RX/RY rotations.
* A `QuantumRegressor` that measures Pauli‑Z on all wires and applies a linear head.
* A top‑level `QuantumHybridRegressor` that stitches the classical backbone and the quantum head together.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchquantum as tq


def generate_superposition_state(
    num_features: int,
    samples: int,
    angle_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a complex superposition state and a sinusoidal label.

    Args:
        num_features: Number of qubits (encoded as 2**num_features dimension).
        samples: Number of samples to generate.
        angle_offset: Phase offset applied to the generated angles.

    Returns:
        Tuple of (states, labels) where states has shape (samples, 2**num_features)
        and labels has shape (samples,).
    """
    omega_0 = np.zeros(2 ** num_features, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_features, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples) + angle_offset
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_features), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that yields either real features or complex states.

    Args:
        samples: Number of samples.
        num_features: Feature dimensionality.
        use_complex: If True, return complex state vectors; otherwise return real features.
    """

    def __init__(self, samples: int, num_features: int, use_complex: bool = False):
        super().__init__()
        self.use_complex = use_complex
        if use_complex:
            self.states, self.labels = generate_superposition_state(num_features, samples)
        else:
            self.features, self.labels = self._generate_real_data(num_features, samples)

    @staticmethod
    def _generate_real_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.use_complex:
            return {"states": torch.tensor(self.states[idx], dtype=torch.cfloat),
                    "target": torch.tensor(self.labels[idx], dtype=torch.float32)}
        else:
            return {"states": torch.tensor(self.features[idx], dtype=torch.float32),
                    "target": torch.tensor(self.labels[idx], dtype=torch.float32)}


class ClassicalEncoder(nn.Module):
    """Linear encoder that maps real features to a complex vector of the same dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.real_weight = nn.Parameter(torch.randn(dim, dim))
        self.real_bias = nn.Parameter(torch.randn(dim))
        self.imag_weight = nn.Parameter(torch.randn(dim, dim))
        self.imag_bias = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = torch.matmul(x, self.real_weight) + self.real_bias
        imag = torch.matmul(x, self.imag_weight) + self.imag_bias
        return torch.complex(real, imag)


class QuantumLayer(tq.QuantumModule):
    """Variational block: random layer + trainable RX/RY on each wire."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)


class QuantumRegressor(tq.QuantumModule):
    """Quantum regression head that measures Pauli‑Z on all wires and applies a linear read‑out."""

    def __init__(self, num_wires: int, shift: float = 0.0):
        super().__init__()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.qlayer = QuantumLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        self.shift = shift

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.qlayer.n_wires,
                                bsz=bsz,
                                device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.qlayer(qdev)
        features = self.measure(qdev)
        return (self.head(features) + self.shift).squeeze(-1)


class QuantumHybridRegressor(nn.Module):
    """Top‑level hybrid regression model."""

    def __init__(
        self,
        num_features: int,
        hidden_dims: list[int] | None = None,
        num_wires: int | None = None,
        use_encoder: bool = True,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, num_wires or hidden_dims[-1]))
        self.classical_backbone = nn.Sequential(*layers)

        self.use_encoder = use_encoder
        self.encoder = ClassicalEncoder(num_features) if use_encoder else None
        self.quantum_head = QuantumRegressor(num_wires or hidden_dims[-1])

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        features = self.classical_backbone(batch)
        states = self.encoder(batch) if self.use_encoder else features
        return self.quantum_head(states)


__all__ = ["RegressionDataset", "QuantumHybridRegressor"]
