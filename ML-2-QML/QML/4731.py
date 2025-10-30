"""
Quantum regression model that mirrors the classical version but replaces
the transformer head with a variational circuit and implements a quantum
convolutional filter.  The implementation uses TorchQuantum and can run on
a simulator or a real QPU.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Dataset utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data where each sample is a 2‑qubit state.
    The labels are a nonlinear function of the superposition angles.
    """
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
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Torch dataset that returns a complex state vector and a float target."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Quantum convolutional filter
# --------------------------------------------------------------------------- #
class QuantumConvFilter(tq.QuantumModule):
    """A simple parametric circuit that acts as a 2‑D filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.circuit = tq.QuantumDevice(n_wires=self.n_qubits)

        # Encode each pixel into a rotation on a qubit
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_qubits)]
        )

        # Trainable rotation per qubit
        self.params = nn.Parameter(torch.randn(self.n_qubits))

    def run(self, data: np.ndarray) -> float:
        """
        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) with values in [0, 1].

        Returns
        -------
        float
            Expectation value of PauliZ on the first qubit, interpreted as a
            probability‑like score.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size)
        # Flatten to a 1‑D array of values
        flat = tensor.view(-1).cpu().numpy()
        # Create a state vector
        state = np.zeros(2 ** self.n_qubits, dtype=complex)
        state[0] = 1.0
        for i, val in enumerate(flat):
            theta = np.pi if val > self.threshold else 0.0
            state = tqf.apply_rx(state, i, theta)
        # Apply trainable rotations
        for i, theta in enumerate(self.params):
            state = tqf.apply_rx(state, i, theta.item())
        # Measure expectation of Z on qubit 0
        probs = tqf.probabilities(state, 0)
        exp_z = probs[0] - probs[1]
        return float(exp_z)

# --------------------------------------------------------------------------- #
# Variational circuit used as the transformer head
# --------------------------------------------------------------------------- #
class VariationalHead(tq.QuantumModule):
    """A small variational circuit that maps a feature vector to a single output."""
    def __init__(self, input_dim: int, n_wires: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.qdev = tq.QuantumDevice(n_wires=n_wires)

        # Simple encoder that maps each input feature into a rotation on a qubit
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i % n_wires]} for i in range(input_dim)]
        )

        # Trainable rotation per qubit
        self.params = nn.Parameter(torch.randn(n_wires))

        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 1) containing the expectation value
            of PauliZ on each qubit, averaged across qubits.
        """
        batch, _ = x.shape
        self.qdev.reset(bsz=batch, device=x.device)

        # Encode input
        self.encoder(self.qdev, x)

        # Apply trainable rotations
        for i, theta in enumerate(self.params):
            self.qdev.rx(theta, wires=i)

        # Measure and average
        out = self.measure(self.qdev)
        return out.mean(dim=1, keepdim=True)

# --------------------------------------------------------------------------- #
# The quantum regression model
# --------------------------------------------------------------------------- #
class QuantumRegressionModel(tq.QuantumModule):
    """
    Quantum regression model that mirrors the classical version but replaces
    the transformer head with a variational circuit.  The rest of the
    architecture is identical to the classical branch.
    """
    def __init__(self,
                 num_features: int,
                 n_wires: int = 8,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 ffn_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Linear(num_features, embed_dim)
        self.transformer = nn.Sequential(
            nn.MultiheadAttention(embed_dim, num_heads,
                                  dropout=dropout,
                                  batch_first=True),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.head = VariationalHead(embed_dim, n_wires=n_wires)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = self.encoder(batch)
        # Transformer block expects (batch, seq, dim)
        attn_out, _ = self.transformer[0](x, x, x)
        x = self.transformer[1](x + attn_out)
        x = self.transformer[2:](x)
        x = self.head(x)
        return x.squeeze(-1)

__all__ = [
    "RegressionDataset",
    "QuantumConvFilter",
    "VariationalHead",
    "QuantumRegressionModel",
    "generate_superposition_data",
]
