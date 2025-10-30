"""Quantum regression with a deep entangled variational ansatz and learnable readout."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from typing import Tuple


def generate_superposition_data(
    num_wires: int,
    samples: int,
    noise_std: float = 0.05,
    mix_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate quantum states of the form
    cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>
    with optional Gaussian noise added to the target labels.

    Parameters
    ----------
    num_wires : int
        Number of qubits in each state.
    samples : int
        Number of samples to generate.
    noise_std : float, default 0.05
        Standard deviation of Gaussian noise added to the labels.
    mix_ratio : float, default 0.5
        Weighting between the clean quantum signal and the noisy component.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``states`` of shape (samples, 2**num_wires) and ``labels`` of shape (samples,).
    """
    # Clean label from the quantum superposition
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)

    # Build the pure states
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = (
            np.cos(thetas[i]) * omega_0
            + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        )

    # Clean labels
    labels_clean = np.sin(2 * thetas) * np.cos(phis)

    # Gaussian noise
    noise = np.random.normal(scale=noise_std, size=samples).astype(np.float32)

    # Mix clean and noisy labels
    labels = (1 - mix_ratio) * labels_clean + mix_ratio * noise
    return states.astype(np.complex64), labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset that returns a dictionary with quantum states and target.
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


class EntangledVariationalLayer(tq.QuantumModule):
    """
    Variational layer that applies a trainable rotation on each qubit followed
    by a layer of CNOTs to induce entanglement. The depth of the ansatz is
    controlled by the ``depth`` parameter.
    """

    def __init__(self, num_wires: int, depth: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.depth = depth
        self.rotation = tq.RX(has_params=True, trainable=True)
        self.entangle = tq.CNOT(has_params=False, trainable=False)

        # Create a list of rotation parameters for each depth layer
        self.params = nn.ParameterList(
            [
                nn.Parameter(torch.randn(num_wires))
                for _ in range(depth)
            ]
        )

    def forward(self, qdev: tq.QuantumDevice) -> None:
        for d in range(self.depth):
            # Apply rotations with trainable parameters
            for wire in range(self.num_wires):
                self.rotation(qdev, wires=wire, params=self.params[d][wire])
            # Entangle adjacent qubits in a ring topology
            for wire in range(self.num_wires):
                target = (wire + 1) % self.num_wires
                self.entangle(qdev, wires=[wire, target])


class QModel(tq.QuantumModule):
    """
    Quantum regression model that encodes classical features onto a quantum
    state, processes them with an entangled variational ansatz, measures
    expectation values, and feeds the result into a learnable linear head.
    """

    def __init__(self, num_wires: int, ansatz_depth: int = 2):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.ansatz = EntangledVariationalLayer(num_wires, depth=ansatz_depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # The head maps the 2**num_wires expectation vector to a single output
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of complex state vectors of shape (B, 2**num_wires).

        Returns
        -------
        torch.Tensor
            Predicted regression values of shape (B,).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode the classical data onto the quantum state
        self.encoder(qdev, state_batch)
        # Apply the variational ansatz
        self.ansatz(qdev)
        # Measure expectation values
        features = self.measure(qdev)
        # Classical readout
        return self.head(features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
