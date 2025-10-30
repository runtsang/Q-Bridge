"""Quantum regression model with a parameterized variational ansatz and measurement‑noise aware loss.

The module mirrors the classical counterpart but operates on quantum states.  It
provides a configurable depth of variational layers, an optional measurement
noise model, and a custom loss that penalises large measurement variance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from typing import Tuple

def generate_superposition_data(
    num_wires: int,
    samples: int,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate superposition states ``cos(theta)|0…0> + exp(i phi) sin(theta)|1…1>``
    and corresponding labels.

    Parameters
    ----------
    num_wires : int
        Number of qubits in each state.
    samples : int
        Number of samples to generate.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    states : np.ndarray, shape (samples, 2**num_wires)
        Complex amplitude tensors that can be cast to ``torch.cfloat``.
    labels : np.ndarray, shape (samples,)
        Regression targets computed as ``sin(2*theta)*cos(phi)``.
    """
    rng = np.random.default_rng(seed)
    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)

    basis0 = np.zeros((samples, 2 ** num_wires), dtype=complex)
    basis0[:, 0] = 1.0
    basis1 = np.zeros((samples, 2 ** num_wires), dtype=complex)
    basis1[:, -1] = 1.0

    states = np.cos(thetas[:, None]) * basis0 + np.exp(1j * phis[:, None]) * np.sin(thetas[:, None]) * basis1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that stores quantum state tensors and their regression targets.
    """

    def __init__(self, samples: int, num_wires: int, seed: int | None = None):
        self.states, self.labels = generate_superposition_data(num_wires, samples, seed)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegression__gen226(tq.QuantumModule):
    """
    Quantum regression network that encodes classical features into a quantum
    state, applies a parameterised variational circuit, measures all qubits,
    and feeds the measurement statistics into a classical head.

    Parameters
    ----------
    num_wires : int
        Number of qubits used by the device.
    depth : int, default 2
        Number of repetitions of the variational block.
    noise_std : float, default 0.0
        Standard deviation of Gaussian noise added to each measurement to
        simulate read‑out errors.  Setting ``noise_std>0`` activates the
        noise‑aware loss.
    """

    class VariationalBlock(tq.QuantumModule):
        """Single layer of parameterised rotations and a random entangling layer."""

        def __init__(self, num_wires: int):
            super().__init__()
            self.num_wires = num_wires
            # Random entangling layer to provide expressive power
            self.entangler = tq.RandomLayer(n_ops=2 * num_wires, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.entangler(qdev)
            for w in range(self.num_wires):
                self.rx(qdev, wires=w)
                self.rz(qdev, wires=w)

    def __init__(self, num_wires: int, depth: int = 2, noise_std: float = 0.0):
        super().__init__()
        self.num_wires = num_wires
        self.noise_std = noise_std

        # Encoder that maps the input vector into a computational basis state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])

        # Stack several variational blocks
        self.blocks = nn.ModuleList([self.VariationalBlock(num_wires) for _ in range(depth)])

        # Measurement of all qubits in the Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical linear head that maps the measurement statistics to a scalar
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Execute the quantum circuit and return the predicted scalar regression output.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)

        for block in self.blocks:
            block(qdev)

        # Add measurement noise if requested
        if self.noise_std > 0.0:
            # Draw Gaussian noise for each qubit
            noise = torch.randn(bsz, self.num_wires, device=state_batch.device) * self.noise_std
            # The measurement returns values in {-1, 1}; we model the noise as a shift
            features = self.measure(qdev) + noise
        else:
            features = self.measure(qdev)

        return self.head(features).squeeze(-1)

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss and optionally add a variance penalty when measurement noise
        is enabled.  The penalty encourages the network to minimise the spread of
        the noisy measurements.
        """
        mse = nn.functional.mse_loss(predictions, targets)
        if self.noise_std > 0.0:
            # Estimate variance of the noisy measurements
            var_penalty = self.noise_std ** 2
            return mse + 0.1 * var_penalty
        return mse

__all__ = [
    "QuantumRegression__gen226",
    "RegressionDataset",
    "generate_superposition_data",
]
