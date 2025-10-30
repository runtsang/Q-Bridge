"""EstimatorQNN__gen304 – Quantum‑classical regression module.

Uses TorchQuantum to encode classical features, a trainable random layer,
measures in the Z basis and feeds the result to a linear head.
Also exposes a Qiskit EstimatorQNN wrapper for hybrid experiments.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset


class QLayer(tq.QuantumModule):
    """Quantum layer that adds randomness and trainable rotations."""

    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)


class EstimatorQNN(tq.QuantumModule):
    """
    Hybrid quantum‑classical regressor.
    Encodes a 2‑dimensional input into n_wires qubits using a Ry encoder,
    processes it with QLayer, measures all qubits in the Z basis,
    and maps the measurement vector to a scalar via a linear head.
    """

    def __init__(self, n_wires: int = 2):
        super().__init__()
        self.n_wires = n_wires
        # Encoder that applies Ry rotations proportional to input features.
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.q_layer = QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, n_wires) – classical features to encode.
        Returns
        -------
        torch.Tensor
            Shape (batch,) – predicted scalar.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate states of the form cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>
    and their labels.  Mirrors the quantum data generator from
    QuantumRegression.py.
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


class RegressionDataset(Dataset):
    """Quantum dataset that yields (state, target) pairs."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return (
            torch.tensor(self.states[idx], dtype=torch.cfloat),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# Optional Qiskit EstimatorQNN wrapper (keeps compatibility with the original seed)
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator

def qiskit_estimator(n_qubits: int = 1) -> QiskitEstimatorQNN:
    """
    Builds a Qiskit EstimatorQNN that encodes a single input
    and uses a trainable rotation as a weight.
    """
    params = [Parameter(f"input{i}") for i in range(n_qubits)]
    weights = [Parameter(f"weight{i}") for i in range(n_qubits)]
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
        qc.ry(params[q], q)
        qc.rx(weights[q], q)
    observable = tq.SparsePauliOp.from_list([("Y" * n_qubits, 1)])
    estimator = Estimator()
    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=params,
        weight_params=weights,
        estimator=estimator,
    )


__all__ = [
    "EstimatorQNN",
    "RegressionDataset",
    "generate_superposition_data",
    "qiskit_estimator",
]
