"""Quantum regression model with configurable variational depth and encoding.

This module extends the original ``QModel`` by allowing the user to choose
between multiple encoding strategies and to stack several variational
layers.  The public interface stays the same so legacy scripts keep
working, while the additional knobs provide a richer quantum
expressivity.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data in the form of |ψ(θ,φ)⟩ = cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩.

    Parameters
    ----------
    num_wires : int
        Number of qubits used to encode the state.
    samples : int
        Number of samples to generate.

    Returns
    -------
    states : np.ndarray of shape (samples, 2**num_wires)
        Complex amplitude vectors.
    labels : np.ndarray of shape (samples,)
        Target values computed as ``sin(2θ) * cos(φ)``.
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


class VariationalBlock(tq.QuantumModule):
    """A single variational layer with parameterised single‑qubit rotations
    followed by a layer of random two‑qubit entanglers.
    """

    def __init__(self, n_wires: int, n_ops: int = 20):
        super().__init__()
        self.n_wires = n_wires
        self.random_entangler = tq.RandomLayer(
            n_ops=n_ops, wires=list(range(n_wires))
        )
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_entangler(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
            self.rz(qdev, wires=wire)


class QModel(tq.QuantumModule):
    """
    Quantum regression network with selectable encoding and depth.

    Parameters
    ----------
    num_wires : int
        Number of qubits used for the variational circuit.
    encoder_name : str, optional
        Key into ``tq.encoder_op_list_name_dict`` for the encoding circuit.
        If ``None`` a simple Ry‑encoding is used.
    n_layers : int, default 1
        Number of stacked variational blocks.
    """

    def __init__(
        self,
        num_wires: int,
        encoder_name: str | None = None,
        n_layers: int = 1,
    ):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[
                encoder_name if encoder_name is not None else f"{num_wires}xRy"
            ]
        )
        self.vblocks = nn.ModuleList(
            [VariationalBlock(num_wires) for _ in range(n_layers)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode the classical data into the quantum state
        self.encoder(qdev, state_batch)
        # Apply the variational layers
        for block in self.vblocks:
            block(qdev)
        # Extract expectation values
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
