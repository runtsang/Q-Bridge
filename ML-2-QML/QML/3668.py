"""Quantum regression module fusing a general encoder, a variational layer,
and a lightweight Estimator‑like circuit.

The model inherits from torchquantum.QuantumModule and accepts a batch of
complex state vectors.  A GeneralEncoder maps the input into the first
`n_wires` qubits, a RandomLayer + RX/RY gates form the core variational
block, and a separate 1‑qubit Estimator circuit injects an additional
trainable feature.  All measurement outcomes are concatenated and fed
through a linear regression head.  This design mirrors the classical
HybridRegressionModel while providing a richer quantum inductive bias.
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum states of the form
    cos(theta)|0...0⟩ + e^{i phi} sin(theta)|1...1⟩.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the state.
    samples : int
        Number of samples to generate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        States of shape (samples, 2**num_wires) and labels of shape
        (samples,).
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
    """Dataset providing complex quantum states and regression targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QLayer(tq.QuantumModule):
    """Core variational block: a RandomLayer followed by RX/RY on every wire."""

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


class EstimatorCircuit(tq.QuantumModule):
    """1‑qubit Estimator‑style circuit (H → RX → RY)."""

    def __init__(self):
        super().__init__()
        self.h = tq.H()
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        # Apply on the last wire (estimator qubit)
        last_wire = qdev.n_wires - 1
        self.h(qdev, wires=[last_wire])
        self.rx(qdev, wires=[last_wire])
        self.ry(qdev, wires=[last_wire])


class HybridRegressionModel(tq.QuantumModule):
    """Combined quantum regression model with an estimator branch."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Main encoder using a Ry‑rotation per wire
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = QLayer(num_wires)
        self.estimator_circuit = EstimatorCircuit()
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Output head: one feature per main wire + the estimator qubit
        self.head = nn.Linear(num_wires + 1, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of complex state vectors, shape (batch, 2**n_wires).

        Returns
        -------
        torch.Tensor
            Predicted regression values, shape (batch,).
        """
        bsz = state_batch.shape[0]
        # One extra wire for the estimator circuit
        qdev = tq.QuantumDevice(n_wires=self.n_wires + 1, bsz=bsz, device=state_batch.device)

        # Encode the input into the first `n_wires` qubits
        self.encoder(qdev, state_batch)

        # Variational block on the main qubits
        self.q_layer(qdev)

        # Estimator circuit on the dedicated qubit
        self.estimator_circuit(qdev)

        # Measure all qubits
        features = self.measure(qdev)  # shape (bsz, n_wires+1)

        return self.head(features).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
