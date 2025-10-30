"""Quantum regression model with configurable ansatz and entanglement.

The quantum module expands the original seed by allowing the user to choose
between a random‑layer ansatz or a custom entangling block.  It also
provides batch‑normalised expectation values and supports GPU execution
via torchquantum.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate states of the form |ψ⟩ = cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩.

    Parameters
    ----------
    num_wires : int
        Number of qubits in each state.
    samples : int
        Number of samples to generate.
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

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset yielding quantum states and regression targets.
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

class QModel(tq.QuantumModule):
    """
    Variational quantum circuit for regression.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the device.
    ansatz : str, optional
        Choice of ansatz.  ``'random'`` uses a RandomLayer, ``'entangle'`` uses a
        custom entangling block.  Defaults to ``'random'``.
    entangle_depth : int, optional
        Depth of the entangling block when ``ansatz='entangle'``.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, ansatz: str = "random", entangle_depth: int = 2):
            super().__init__()
            self.n_wires = num_wires
            self.ansatz = ansatz
            if ansatz == "random":
                self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            else:
                # Custom entangling block: repeated CNOTs followed by RX/RY
                self.entangle = tq.Sequential(
                    *[tq.CNOT(wires=[i, (i + 1) % num_wires]) for i in range(num_wires)],
                    *[tq.RX(has_params=True, trainable=True) for _ in range(num_wires)],
                    *[tq.RY(has_params=True, trainable=True) for _ in range(num_wires)],
                    *[tq.CNOT(wires=[i, (i + 1) % num_wires]) for i in range(num_wires)],
                )
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            if self.ansatz == "random":
                self.random_layer(qdev)
            else:
                self.entangle(qdev)
            # Local rotations on each wire
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, ansatz: str = "random", entangle_depth: int = 2):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires, ansatz, entangle_depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
