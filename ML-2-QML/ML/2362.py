"""Hybrid sampler‑regression model combining classical sampling and quantum regression.

This module builds on the simple 2‑input sampler and the quantum regression
network.  The sampler produces a probability distribution over a 2^n‑dimensional
basis, which is then used as the input state for the quantum regression head.
The design allows a smooth transition from a purely classical network to a
hybrid one by toggling the `use_quantum` flag.

The class can be instantiated with any number of features and wires, making
it suitable for a wide range of regression tasks on quantum‑encoded data.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data in the form
    cos(θ)|0…0⟩ + e^{iϕ} sin(θ)|1…1⟩.

    The target is sin(2θ) cos(ϕ).  This mirrors the quantum regression
    reference and provides a realistic benchmark for the hybrid model.
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
    """Dataset wrapping the synthetic superposition states."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridSamplerRegression(nn.Module):
    """Classical hybrid sampler‑regression network.

    Parameters
    ----------
    num_features : int
        Dimensionality of the classical input vector.
    num_wires : int
        Number of qubits used for the regression head.
    use_quantum : bool, default=True
        If True, the regression head is a quantum module.  Otherwise a
        purely classical linear head is used.
    """

    def __init__(self, num_features: int, num_wires: int, use_quantum: bool = True):
        super().__init__()
        self.num_wires = num_wires
        self.use_quantum = use_quantum

        # Sampler: maps classical features to a probability vector over 2^n basis states
        self.sampler = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 2 ** num_wires),
        )

        if use_quantum:
            # Quantum regression head
            from torchquantum import QuantumModule, QuantumDevice, MeasureAll, PauliZ, RandomLayer, RX, RY, GeneralEncoder
            import torch.nn as nn

            class QLayer(QuantumModule):
                def __init__(self, n_wires: int):
                    super().__init__()
                    self.n_wires = n_wires
                    self.random_layer = RandomLayer(n_ops=30, wires=list(range(n_wires)))
                    self.rx = RX(has_params=True, trainable=True)
                    self.ry = RY(has_params=True, trainable=True)

                def forward(self, qdev: QuantumDevice):
                    self.random_layer(qdev)
                    for wire in range(self.n_wires):
                        self.rx(qdev, wires=wire)
                        self.ry(qdev, wires=wire)

            self.encoder = GeneralEncoder(GeneralEncoder.encoder_op_list_name_dict[f"{num_wires}xRy"])
            self.q_layer = QLayer(num_wires)
            self.measure = MeasureAll(PauliZ)
            self.head = nn.Linear(num_wires, 1)
        else:
            # Classical regression head as a fallback
            self.head = nn.Sequential(
                nn.Linear(2 ** num_wires, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch,).
        """
        probs = F.softmax(self.sampler(x), dim=-1)  # (batch, 2^n)
        if self.use_quantum:
            bsz = probs.shape[0]
            qdev = torchquantum.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=probs.device)
            self.encoder(qdev, probs)
            self.q_layer(qdev)
            features = self.measure(qdev)
            return self.head(features).squeeze(-1)
        else:
            return self.head(probs).squeeze(-1)


__all__ = ["HybridSamplerRegression", "RegressionDataset", "generate_superposition_data"]
