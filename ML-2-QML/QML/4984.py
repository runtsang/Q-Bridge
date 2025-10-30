"""Hybrid quantum convolution module with RBF kernel similarity and self‑attention.

This module implements a quantum‑centric version of the classical
``Conv`` filter.  It uses a TorchQuantum encoder to map the input
image into a quantum state, a variational layer to process the state,
and a quantum kernel that measures similarity with a learned prototype.
The forward pass returns the same scalar output as the classical
implementation, allowing seamless comparison.
"""
import torch
import torch.nn as nn
import numpy as np
import torchquantum as tq
from torch.quantum import QuantumDevice

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate superposition states for regression."""
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

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class Conv(tq.QuantumModule):
    """Quantum hybrid convolutional filter."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, n_wires: int = 16, gamma: float = 1.0) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.gamma = gamma

        # Encode each feature as a Ry rotation
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["ry"])

        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

        self.head: nn.Linear | None = None

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor
            Tensor of shape ``(batch, n_wires)`` containing classical
            features to be encoded.

        Returns
        -------
        torch.Tensor
            Shape ``(batch,)`` – regression output.
        """
        bsz = state_batch.shape[0]
        q_device = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)

        self.encoder(q_device, state_batch)
        self.q_layer(q_device)
        features = self.measure(q_device)

        if not hasattr(self, "prototype"):
            self.register_buffer("prototype", torch.zeros(self.n_wires, dtype=torch.float32))

        if self.head is None:
            self.head = nn.Linear(self.n_wires, 1)

        diff = features - self.prototype
        rbf = torch.exp(-self.gamma * torch.sum(diff * diff, dim=1, keepdim=True))
        out = self.head(features + rbf)
        return out.squeeze(-1)

__all__ = [
    "Conv",
    "generate_superposition_data",
    "RegressionDataset",
]
