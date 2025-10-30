"""Hybrid regression model combining quantum feature extraction with a classical head.

The quantum part uses a general encoder to map classical inputs to a quantum state,
followed by a parameterised quantum fully‑connected layer (QFC) that implements
a small variational circuit. The measurement results are fed into a linear
classifier to produce the regression output.

This module builds on the original QuantumRegression example and integrates
the fully‑connected quantum layer concept from the FCL reference.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate superposition states of the form
    cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.

    Parameters
    ----------
    num_wires: int
        Number of qubits in each state.
    samples: int
        Number of states to generate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        States of shape (samples, 2**num_wires) and labels.
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
    """
    PyTorch dataset wrapping the quantum superposition data.
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

class QuantumFullyConnectedLayer(tq.QuantumModule):
    """
    Parameterised quantum circuit that mimics a classical fully‑connected layer.
    Each qubit receives a trainable Ry rotation after a small random feature
    layer. The circuit outputs the expectation value of Pauli‑Z on each qubit.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        # Random feature layer to enrich the state
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
        # Trainable Ry gates
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        # Inject random features
        self.random_layer(qdev)
        # Apply a trainable Ry to each qubit
        for wire in range(self.n_wires):
            self.ry(qdev, wires=wire)

class HybridRegressionModel(tq.QuantumModule):
    """
    Quantum‑classical hybrid regression model.

    The model consists of:
      1. GeneralEncoder – maps classical input to a quantum state.
      2. QuantumFullyConnectedLayer – variational circuit producing
         a feature vector of Pauli‑Z expectations.
      3. MeasureAll – extracts the feature vector.
      4. Classical linear head – maps to a scalar output.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that applies a Ry rotation per qubit based on the input
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Quantum fully‑connected layer
        self.qfc = QuantumFullyConnectedLayer(num_wires)
        # Measurement of all qubits
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch: torch.Tensor
            Batch of input states of shape (batch, 2**num_wires).

        Returns
        -------
        torch.Tensor
            Regression prediction of shape (batch,).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode classical data into the quantum device
        self.encoder(qdev, state_batch)
        # Apply the quantum fully‑connected layer
        self.qfc(qdev)
        # Extract expectation values
        features = self.measure(qdev)
        # Classical linear head
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
