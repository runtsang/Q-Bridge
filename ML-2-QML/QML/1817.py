"""Quantum regression model and dataset with a configurable variational circuit.

The module mirrors the classical interface but replaces the neural network with
a parameterised quantum circuit that is trained end‑to‑end with PyTorch.
Features
--------
- Flexible circuit depth and entanglement pattern.
- General encoder that can ingest arbitrary feature vectors.
- Expectation‑value readout on all qubits followed by a classical linear head.
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset


def generate_superposition_data(
    num_wires: int,
    samples: int,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create quantum states of the form ``cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>``.

    Parameters
    ----------
    num_wires : int
        Number of qubits in each state.
    samples : int
        Number of states to generate.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the label.

    Returns
    -------
    states : ndarray of shape (samples, 2**num_wires)
        Complex state vectors.
    labels : ndarray of shape (samples,)
        Target values based on ``sin(2*theta)*cos(phi)``.
    """
    omega_0 = np.zeros(2**num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2**num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    if noise_std > 0.0:
        labels += np.random.normal(scale=noise_std, size=labels.shape).astype(np.float32)
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Torch ``Dataset`` that returns a complex state vector and a scalar target.

    Parameters
    ----------
    samples : int
        Number of samples in the dataset.
    num_wires : int
        Size of the quantum register.
    """

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)
        self.states = self.states.astype(np.complex64)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegressionModel(tq.QuantumModule):
    """
    Variational quantum circuit for regression.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the device.
    depth : int, default 2
        Number of entangling layers in the variational circuit.
    entangler_map : Sequence[tuple[int, int]], optional
        Custom entanglement pattern.  If ``None`` a linear chain is used.
    """

    class VariationalLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, depth: int, entangler_map: list[tuple[int, int]] | None):
            super().__init__()
            self.num_wires = num_wires
            self.depth = depth
            self.entangler_map = entangler_map or [(i, i + 1) for i in range(num_wires - 1)]
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.cz = tq.CZ(has_params=False, trainable=False)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for _ in range(self.depth):
                for wire in range(self.num_wires):
                    self.rx(qdev, wires=wire)
                    self.ry(qdev, wires=wire)
                for ctrl, tgt in self.entangler_map:
                    self.cz(qdev, wires=[ctrl, tgt])

    def __init__(self, num_wires: int, depth: int = 2, entangler_map: list[tuple[int, int]] | None = None):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.var_layer = self.VariationalLayer(num_wires, depth, entangler_map)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.var_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
