"""Advanced quantum regression model with entangling variational layers.

The quantum architecture extends the original seed by introducing an
entangling block (CX & CZ gates) and a deeper variational ansatz.  The
encoder is still a general Ry rotation, but the feature extraction now
captures multi‑qubit correlations.  The final readout uses a linear
head on the expectation values of Pauli‑Z, matching the dimensionality
of the classical model.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int,
                                noise_scale: float = 0.0,
                                seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic superposition data in the computational basis.

    Parameters
    ----------
    num_wires : int
        Number of qubits (feature dimensionality).
    samples : int
        Number of samples to generate.
    noise_scale : float, default 0.0
        Standard deviation of Gaussian noise added to the labels.
    seed : int | None, default None
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        State vectors and corresponding labels.
    """
    if seed is not None:
        np.random.seed(seed)
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
    if noise_scale > 0.0:
        labels += noise_scale * np.random.randn(samples)
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for the quantum superposition data.
    """
    def __init__(self, samples: int, num_wires: int,
                 noise_scale: float = 0.0, seed: int | None = None):
        self.states, self.labels = generate_superposition_data(
            num_wires, samples, noise_scale=noise_scale, seed=seed
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegressionModel(tq.QuantumModule):
    """
    Variational quantum regression model with entangling layers.
    """
    class QLayer(tq.QuantumModule):
        """
        Parameterised entangling ansatz: repeated layers of CX, RX, RY, CZ.
        """
        def __init__(self, num_wires: int, num_layers: int = 3):
            super().__init__()
            self.num_wires = num_wires
            self.num_layers = num_layers
            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                # Entangling block
                self.layers.append(tq.CNOT(has_params=False))
                self.layers.append(tq.CZ(has_params=False))
                # Parameterised rotations
                self.layers.append(tq.RX(has_params=True, trainable=True))
                self.layers.append(tq.RY(has_params=True, trainable=True))

        def forward(self, qdev: tq.QuantumDevice):
            for layer in self.layers:
                layer(qdev)

    def __init__(self, num_wires: int, num_layers: int = 3):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = self.QLayer(num_wires, num_layers=num_layers)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
