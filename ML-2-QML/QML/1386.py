"""Quantum regression module with multi‑encoder support and trainable readout.

The module extends the original seed by allowing the user to choose among
different state‑encoding strategies (Hadamard, Ry, or a custom circuit),
and by measuring both Pauli‑Z and Pauli‑X expectations.  The readout
layer is a learnable linear map that can optionally be fused with the
final prediction head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from typing import Tuple, Dict

def generate_superposition_data(
    num_wires: int,
    samples: int,
    *,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate superposition states and labels.

    Parameters
    ----------
    num_wires : int
        Number of qubits per sample.
    samples : int
        Number of samples to generate.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    states : np.ndarray of shape (samples, 2**num_wires)
        Complex amplitudes of the generated states.
    labels : np.ndarray of shape (samples,)
        Target values.
    """
    rng = np.random.default_rng(seed)
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapping quantum states and real‑valued targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """Variational quantum regression model with configurable encoder."""

    class QLayer(tq.QuantumModule):
        """Parameterized circuit applied after encoding."""

        def __init__(self, num_wires: int, n_ops: int = 30):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(
        self,
        num_wires: int,
        encoder_name: str = "Ry",
        measure_pauli: Tuple[str,...] = ("Z",),
        readout_dim: int | None = None,
    ):
        super().__init__()
        self.n_wires = num_wires
        self.encoder_name = encoder_name
        self.measure_pauli = measure_pauli

        # Encoder registry
        self.encoders: Dict[str, tq.QuantumModule] = {
            "Hadamard": tq.Hadamard,
            "Ry": tq.RY,
            "Rx": tq.RX,
            "Rz": tq.RZ,
        }
        if encoder_name not in self.encoders:
            raise ValueError(f"Unsupported encoder '{encoder_name}'. "
                             f"Supported: {list(self.encoders.keys())}")
        self.encoder = tq.GeneralEncoder(
            [self.encoders[encoder_name]], wires=list(range(num_wires))
        )

        self.q_layer = self.QLayer(num_wires)
        # Measure all chosen Pauli terms
        self.measure = tq.MeasureAll(
            *[getattr(tq, f"Pauli{p}") for p in measure_pauli]
        )
        # Readout: linear mapping from measurement vector to a latent vector
        feature_dim = len(measure_pauli) * num_wires
        if readout_dim is None:
            readout_dim = num_wires
        self.readout = nn.Linear(feature_dim, readout_dim)
        self.head = nn.Linear(readout_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        # Flatten measurement tensor: (bsz, num_wires, num_pauli)
        features = features.view(bsz, -1)
        latent = self.readout(features)
        return self.head(latent).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
