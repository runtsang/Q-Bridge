"""Quantum regression model with a controlled‑phase variational ansatz.

The module mirrors the classical counterpart but replaces the residual
network with a quantum circuit that contains a random layer, single‑qubit
rotations, a controlled‑phase block, and a measurement head that outputs
multiple regression targets.  The dataset generator produces multi‑output
targets and the model can output multiple regression values by concatenating
several measurement features.  The quantum layer is built using torchquantum
and can be executed on a CPU or GPU backend.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset


def generate_superposition_data(
    num_wires: int,
    samples: int,
    output_dim: int = 1,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a quantum superposition dataset with multi‑output targets.

    Each example is a state ``cos(theta)|0...0> + exp(i phi) sin(theta)|1...1>``.
    The target vector contains ``output_dim`` sinusoidal values that depend on
    ``theta`` and random phases.
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

    outputs = np.zeros((samples, output_dim), dtype=np.float32)
    for i in range(output_dim):
        phi_offset = rng.uniform(0, 2 * np.pi, size=samples)
        outputs[:, i] = np.sin(2 * thetas) * np.cos(phi_offset)

    return states, outputs.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns quantum state vectors and multi‑output targets."""

    def __init__(
        self,
        samples: int,
        num_wires: int,
        output_dim: int = 1,
        *,
        seed: int | None = None,
    ):
        self.states, self.labels = generate_superposition_data(
            num_wires, samples, output_dim=output_dim, seed=seed
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridRegressionModel(tq.QuantumModule):
    """Quantum regression model with a controlled‑phase ansatz and variational head.

    The circuit contains a random layer, single‑qubit rotations, a
    controlled‑phase block, and a measurement head that outputs a vector
    of length ``output_dim``.  The final classical head maps the vector
    to regression targets.
    """

    class QLayer(tq.QuantumModule):
        """Core variational layer with controlled‑phase gates."""

        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.cphase = tq.CPhase(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                self.cphase(qdev, wires=[wire, wire + 1])

    def __init__(self, num_wires: int, output_dim: int = 1):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, output_dim)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
