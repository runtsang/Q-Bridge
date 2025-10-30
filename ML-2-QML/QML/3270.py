"""Hybrid fully connected layer and regression model for quantum training.

This module fuses the variational quantum circuit from the original FCL seed
with the quantum regression architecture of the second seed.  The
:class:`HybridFCL` class inherits from :class:`torchquantum.QuantumModule`,
provides a ``forward`` method that maps a batch of complex state vectors to
a scalar prediction, and a ``run`` method that evaluates a simple 1‑qubit
parameterised circuit using TorchQuantum primitives.
"""

from __future__ import annotations

from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a toy quantum regression dataset.

    Parameters
    ----------
    num_wires : int
        Number of qubits in each state vector.
    samples : int
        Number of samples to generate.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``states`` of shape ``(samples, 2**num_wires)`` (complex) and
        corresponding ``labels``.
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
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapping the synthetic quantum superposition data."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridFCL(tq.QuantumModule):
    """
    Quantum hybrid fully‑connected layer.

    The network first encodes a complex state vector using a pre‑defined
    parameterised circuit, then applies a trainable variational layer
    followed by measurement and a classical linear head.  The ``run`` method
    evaluates a minimal 1‑qubit circuit consisting of an H gate and a
    parameterised Ry rotation, returning the expectation value of Z.
    """

    class QLayer(tq.QuantumModule):
        """Variational layer used inside the regression network."""

        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int = 2):
        super().__init__()
        self.n_wires = num_wires
        # Encoder uses a fixed Ry‑based circuit appropriate for the wire count.
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Map a batch of complex state vectors to a scalar prediction."""
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate a simple 1‑qubit parameterised circuit.

        The circuit consists of an H gate followed by a Ry rotation
        parameterised by ``thetas``.  The expectation value of Pauli‑Z
        is returned for each angle.
        """
        bsz = len(thetas)
        dev = tq.QuantumDevice(n_wires=1, bsz=bsz, device="cpu")
        # Apply H to all wires
        dev.apply(tq.H(), wires=0)
        # Apply Ry with parameterised angles
        for idx, theta in enumerate(thetas):
            dev.apply(tq.RY(theta), wires=0, batch_slice=idx)
        # Measure expectation of Z
        exp = dev.expectation(tq.PauliZ, wires=0)
        return exp.detach().cpu().numpy()

__all__ = ["HybridFCL", "RegressionDataset", "generate_superposition_data"]
