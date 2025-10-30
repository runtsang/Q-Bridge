"""Quantum regression model with adaptive measurement and learnable head.

The quantum module encodes classical data into a superposition state,
runs a parameterised variational circuit, measures in the Pauli‑Z basis,
and feeds the expectation values into a linear head.  The data generator
produces complex state vectors with optional label noise and a fixed
random seed for reproducibility.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset
from typing import Tuple, Optional

def generate_superposition_data(
    num_wires: int,
    samples: int,
    *,
    noise_std: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset of quantum states.

    Parameters
    ----------
    num_wires : int
        Number of qubits in each state.
    samples : int
        Number of samples to generate.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the target.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    states : np.ndarray of shape (samples, 2**num_wires)
        Complex state vectors.
    labels : np.ndarray of shape (samples,)
        Target values.
    """
    rng = np.random.default_rng(random_state)
    # Basis states |0...0> and |1...1>
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = (
            np.cos(thetas[i]) * omega_0
            + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        )
    labels = np.sin(2 * thetas) * np.cos(phis)
    if noise_std is not None:
        labels += rng.normal(scale=noise_std, size=labels.shape)
    return states.astype(np.complex64), labels.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Torch dataset for the synthetic quantum regression data.
    """

    def __init__(
        self,
        samples: int,
        num_wires: int,
        *,
        noise_std: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        self.states, self.labels = generate_superposition_data(
            num_wires, samples, noise_std=noise_std, random_state=random_state
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegression__gen172(tq.QuantumModule):
    """
    Quantum variational regression model with adaptive measurement
    and a learnable linear head.
    """

    class _VariationalBlock(tq.QuantumModule):
        """
        Parameterised circuit consisting of a random layer followed by
        trainable single‑qubit rotations.
        """

        def __init__(self, n_wires: int, n_ops: int = 20):
            super().__init__()
            self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(qdev.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(
        self,
        num_wires: int,
        n_layers: int = 3,
        n_ops: int = 20,
    ):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps classical data into a superposition state
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        # Stack of variational blocks
        self.var_blocks = nn.ModuleList(
            [self._VariationalBlock(num_wires, n_ops=n_ops) for _ in range(n_layers)]
        )
        # Adaptive measurement: measure all qubits in the Pauli‑Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical linear head that maps expectation values to a scalar
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the quantum regression model.

        Parameters
        ----------
        state_batch : torch.Tensor of shape (batch, 2**num_wires)
            Batch of classical feature vectors that will be encoded.

        Returns
        -------
        torch.Tensor of shape (batch,)
            Predicted target values.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=state_batch.device
        )
        self.encoder(qdev, state_batch)
        for block in self.var_blocks:
            block(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QuantumRegression__gen172", "RegressionDataset", "generate_superposition_data"]
