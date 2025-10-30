"""Quantum regression model with a variational circuit.

The quantum module encodes real‑valued inputs into a parameterised state,
passes the state through a randomised variational layer with trainable
RX/RY rotations, measures all qubits in the Pauli‑Z basis, and feeds the
expectation values into a classical linear head.  The data generator
produces superposition states that can be used directly as input.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset
from typing import Tuple

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]

def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a batch of states of the form
        cos(θ)|0…0⟩ + exp(iφ) sin(θ)|1…1⟩
    and corresponding regression targets.
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

class RegressionDataset(Dataset):
    """
    Dataset that returns a dictionary with keys ``states`` (complex) and
    ``target`` (float).  The ``states`` tensor can be fed directly into
    the quantum encoder.
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
    Quantum regression model.  Encodes input amplitudes using a
    pre‑built Ry‑rotation encoder, runs a variational circuit with
    random gates and trainable single‑qubit rotations, measures all
    qubits in the Z basis, and passes the expectation values to a
    classical linear head.
    """
    class Encoder(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.num_wires = num_wires
            # Use the built‑in Ry‑encoding for the given wire count
            self.encoder_ops = tq.encoder_op_list_name_dict[f"{num_wires}xRy"]

        def forward(self, qdev: tq.QuantumDevice, input_states: torch.Tensor):
            # input_states shape: (batch, 2**num_wires)
            self.encoder_ops(qdev, input_states)

    class VariationalLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.num_wires = num_wires
            # Random layer to initialise trainable parameters
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
            # Trainable single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.num_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = self.Encoder(num_wires)
        self.var_layer = self.VariationalLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of complex state vectors of shape (batch, 2**n_wires).

        Returns
        -------
        torch.Tensor
            Predicted target values of shape (batch,).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.var_layer(qdev)
        features = self.measure(qdev)  # shape: (batch, n_wires)
        return self.head(features).squeeze(-1)
