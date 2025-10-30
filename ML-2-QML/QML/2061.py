"""Quantum regression with an adaptive measurement layer.

The implementation extends the original seed by adding a depth‑controlled
variational circuit and a learnable measurement basis. The public API
mirrors the original: ``RegressionDataset`` and ``QModel`` are exported.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from typing import Tuple, Optional


def generate_superposition_data(
    num_wires: int,
    samples: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create data in the form |ψ(θ,φ)⟩ = cosθ|0…0⟩ + e^{iφ}sinθ|1…1⟩.

    Parameters
    ----------
    num_wires : int
        Number of qubits in each state.
    samples : int
        Number of samples to generate.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        States array of shape (samples, 2**num_wires) and labels array of
        shape (samples,).
    """
    rng = rng or np.random.default_rng(42)
    omega_0 = np.zeros(2**num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2**num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)
    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset compatible with the classical counterpart.

    The ``__getitem__`` method returns a dictionary with keys ``states`` and
    ``target``.  ``states`` are complex tensors suitable for
    ``torchquantum.QuantumDevice``.
    """

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class AdaptiveQLayer(tq.QuantumModule):
    """
    A depth‑controlled variational layer that ends with a learnable Pauli‑X
    rotation before measurement.

    Parameters
    ----------
    num_wires : int
        Number of qubits.
    depth : int, default 3
        Number of repetitions of random and parameterized rotations.
    """

    def __init__(self, num_wires: int, depth: int = 3):
        super().__init__()
        self.num_wires = num_wires
        self.depth = depth
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.adapt_rx = tq.RX(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for _ in range(self.depth):
            for wire in range(self.num_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
        for wire in range(self.num_wires):
            self.adapt_rx(qdev, wires=wire)


class QModel(tq.QuantumModule):
    """
    Main quantum regression model.

    The encoder uses a generalized Ry encoding; the variational layer
    `AdaptiveQLayer` learns measurement bases; the head is a linear map
    from the expectation values to a scalar prediction.
    """

    def __init__(self, num_wires: int, depth: int = 3):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = AdaptiveQLayer(num_wires, depth=depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
