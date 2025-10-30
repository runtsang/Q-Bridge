"""Quantum regression model with variational circuit, entanglement, and measurement pooling."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int, noise_std: float = 0.05, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data in the Hilbert space of ``num_wires`` qubits.
    
    Parameters
    ----------
    num_wires : int
        Number of qubits.
    samples : int
        Number of samples.
    noise_std : float, default 0.05
        Gaussian noise added to labels.
    random_state : int | None
        Seed for reproducibility.
    
    Returns
    -------
    states : np.ndarray
        State vectors of shape (samples, 2**num_wires).
    labels : np.ndarray
        Target values of shape (samples,).
    """
    rng = np.random.default_rng(random_state)
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
    labels += rng.normal(scale=noise_std, size=samples)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset for quantum regression.
    """
    def __init__(self, samples: int, num_wires: int, *, noise_std: float = 0.05, random_state: int | None = None):
        self.states, self.labels = generate_superposition_data(num_wires, samples, noise_std, random_state)
    
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)
    
    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """
    Variational quantum regression model.
    
    The circuit consists of an amplitude‑encoding layer, followed by
    alternating layers of parameterised rotations and entangling gates.
    A measurement‑pooling layer collapses the many‑body state to a
    feature vector that is fed into a tiny classical read‑out network.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, n_layers: int = 2):
            super().__init__()
            self.n_wires = n_wires
            self.n_layers = n_layers
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.cnot = tq.CNOT(has_params=False, trainable=False, wires=None)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for _ in range(self.n_layers):
                for w in range(self.n_wires):
                    self.rx(qdev, wires=w)
                    self.ry(qdev, wires=w)
                for w in range(self.n_wires):
                    next_w = (w + 1) % self.n_wires
                    self.cnot(qdev, wires=[w, next_w])

    def __init__(self, num_wires: int, n_layers: int = 2):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires, n_layers)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.readout = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.readout(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
