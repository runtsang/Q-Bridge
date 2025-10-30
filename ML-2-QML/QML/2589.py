"""Hybrid quantum regression model.

The quantum module encodes classical features into a parameterised
circuit, applies a random entangling layer, measures all qubits, and
feeds the expectation values into a classical head.  The structure
mirrors the classical model so that the two can be swapped
directly.  The `HybridRegressionModel` class is fully compatible
with PyTorch training loops.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic quantum superposition states and labels.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the device.
    samples : int
        Number of samples to generate.

    Returns
    -------
    states : np.ndarray
        Array of shape (samples, 2**num_wires) with complex amplitudes.
    labels : np.ndarray
        Target vector of shape (samples,).
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    rng = np.random.default_rng()
    thetas = rng.uniform(0, 2 * np.pi, size=samples)
    phis = rng.uniform(0, 2 * np.pi, size=samples)

    states = np.cos(thetas)[:, None] * omega_0 + np.exp(1j * phis)[:, None] * np.sin(thetas)[:, None] * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapper for the quantum superposition data."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridRegressionModel(tq.QuantumModule):
    """
    Quantum encoder + classical head for regression.

    The encoder uses a parameterised general encoding that maps the
    input vector to a superposition.  A random entangling layer
    (`RandomLayer`) is applied, followed by a trainable rotation
    layer on each qubit.  After measuring all qubits in the Pauli‑Z
    basis, the expectation values are passed through a dropout‑regularised
    linear head to produce the final scalar prediction.
    """

    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            # Trainable single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, dropout: float = 0.1):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps a real vector to a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self._QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head identical to the ML version
        self.head = nn.Sequential(
            nn.Linear(num_wires, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        states : torch.Tensor
            Tensor of shape (batch, 2**num_wires) containing the
            complex amplitudes of the input states.
        """
        bsz = states.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=states.device)
        self.encoder(qdev, states)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
